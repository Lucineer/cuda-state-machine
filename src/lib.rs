/*!
# cuda-state-machine

Finite state machines for agent behavior.

Agents transition between states — idle, working, resting, alert.
This crate provides FSM construction with guards, actions, and
hierarchical nesting.

- States and transitions
- Guard conditions (conditional transitions)
- Entry/exit actions
- Hierarchical states (nested machines)
- State history
- Serialization
*/

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A state
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct State {
    pub id: String,
    pub parent: Option<String>,  // for hierarchical
    pub on_entry: Option<String>,
    pub on_exit: Option<String>,
    pub is_final: bool,
}

/// A transition
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transition {
    pub from: String,
    pub to: String,
    pub event: String,
    pub guard: Option<String>,      // condition expression
    pub action: Option<String>,     // side effect
    pub priority: u32,
}

/// State change record
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateChange {
    pub from: String,
    pub to: String,
    pub event: String,
    pub timestamp: u64,
    pub action_performed: Option<String>,
}

/// A finite state machine
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateMachine {
    pub name: String,
    pub initial_state: String,
    pub states: HashMap<String, State>,
    pub transitions: Vec<Transition>,
    pub current_state: String,
    pub history: Vec<StateChange>,
    pub event_counts: HashMap<String, u64>,
}

impl StateMachine {
    pub fn new(name: &str, initial: &str) -> Self {
        let mut sm = StateMachine { name: name.to_string(), initial_state: initial.to_string(), states: HashMap::new(), transitions: vec![], current_state: initial.to_string(), history: vec![], event_counts: HashMap::new() };
        sm.states.insert(initial.to_string(), State { id: initial.to_string(), parent: None, on_entry: None, on_exit: None, is_final: false });
        sm
    }

    /// Add a state
    pub fn add_state(&mut self, id: &str, parent: Option<&str>, is_final: bool) {
        self.states.insert(id.to_string(), State { id: id.to_string(), parent: parent.map(|p| p.to_string()), on_entry: None, on_exit: None, is_final });
    }

    /// Add a transition
    pub fn add_transition(&mut self, from: &str, to: &str, event: &str) {
        self.transitions.push(Transition { from: from.to_string(), to: to.to_string(), event: event.to_string(), guard: None, action: None, priority: 0 });
    }

    /// Add a guarded transition
    pub fn add_guarded(&mut self, from: &str, to: &str, event: &str, guard: &str) {
        self.transitions.push(Transition { from: from.to_string(), to: to.to_string(), event: event.to_string(), guard: Some(guard.to_string()), action: None, priority: 1 });
    }

    /// Set entry/exit actions
    pub fn on_entry(&mut self, state: &str, action: &str) {
        if let Some(s) = self.states.get_mut(state) { s.on_entry = Some(action.to_string()); }
    }
    pub fn on_exit(&mut self, state: &str, action: &str) {
        if let Some(s) = self.states.get_mut(state) { s.on_exit = Some(action.to_string()); }
    }

    /// Process an event — returns true if transition occurred
    pub fn handle(&mut self, event: &str, context: &HashMap<String, String>) -> bool {
        *self.event_counts.entry(event.to_string()).or_insert(0) += 1;
        // Find matching transitions (sorted by priority desc)
        let mut matching: Vec<&Transition> = self.transitions.iter()
            .filter(|t| t.from == self.current_state && t.event == event)
            .filter(|t| t.guard.as_ref().map_or(true, |g| self.eval_guard(g, context)))
            .collect();
        matching.sort_by(|a, b| b.priority.cmp(&a.priority));

        if let Some(transition) = matching.first() {
            let from = self.current_state.clone();
            let action = transition.action.clone();
            // Exit action
            if let Some(state) = self.states.get(&from) {
                if let Some(ref exit) = state.on_exit { /* execute exit action */ }
            }
            // Transition
            self.current_state = transition.to.clone();
            // Entry action
            if let Some(state) = self.states.get(&transition.to) {
                if let Some(ref entry) = state.on_entry { /* execute entry action */ }
            }
            self.history.push(StateChange { from, to: transition.to.clone(), event: event.to_string(), timestamp: now(), action_performed: action });
            return true;
        }
        false
    }

    /// Evaluate a guard condition (simple key=value checks)
    fn eval_guard(&self, guard: &str, context: &HashMap<String, String>) -> bool {
        if let Some(pos) = guard.find("==") {
            let key = guard[..pos].trim();
            let val = guard[pos+2..].trim();
            context.get(key).map_or(false, |v| v == val)
        } else if let Some(pos) = guard.find("!=") {
            let key = guard[..pos].trim();
            let val = guard[pos+2..].trim();
            context.get(key).map_or(true, |v| v != val)
        } else { true }
    }

    /// Get available events from current state
    pub fn available_events(&self) -> Vec<&str> {
        let mut events: Vec<&str> = self.transitions.iter()
            .filter(|t| t.from == self.current_state)
            .map(|t| t.event.as_str())
            .collect();
        events.sort();
        events.dedup();
        events
    }

    /// Is the machine in a final state?
    pub fn is_finished(&self) -> bool {
        self.states.get(&self.current_state).map_or(false, |s| s.is_final)
    }

    /// Get state path (ancestors)
    pub fn state_path(&self) -> Vec<String> {
        let mut path = vec![self.current_state.clone()];
        let mut current = self.current_state.clone();
        while let Some(state) = self.states.get(&current) {
            if let Some(ref parent) = state.parent {
                path.push(parent.clone());
                current = parent.clone();
            } else { break; }
        }
        path.reverse();
        path
    }

    /// Summary
    pub fn summary(&self) -> String {
        let transitions_from = self.transitions.iter().filter(|t| t.from == self.current_state).count();
        format!("FSM[{}]: state={}, states={}, transitions={}, events={}, history={}",
            self.name, self.current_state, self.states.len(), self.transitions.len(),
            self.event_counts.len(), self.history.len())
    }
}

fn now() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_traffic_light() -> StateMachine {
        let mut sm = StateMachine::new("traffic", "green");
        sm.add_state("yellow", None, false);
        sm.add_state("red", None, false);
        sm.add_transition("green", "yellow", "timer");
        sm.add_transition("yellow", "red", "timer");
        sm.add_transition("red", "green", "timer");
        sm
    }

    #[test]
    fn test_basic_transition() {
        let mut sm = make_traffic_light();
        assert_eq!(sm.current_state, "green");
        sm.handle("timer", &HashMap::new());
        assert_eq!(sm.current_state, "yellow");
        sm.handle("timer", &HashMap::new());
        assert_eq!(sm.current_state, "red");
    }

    #[test]
    fn test_invalid_event() {
        let mut sm = make_traffic_light();
        let handled = sm.handle("unknown", &HashMap::new());
        assert!(!handled);
        assert_eq!(sm.current_state, "green");
    }

    #[test]
    fn test_guarded_transition() {
        let mut sm = StateMachine::new("door", "closed");
        sm.add_state("open", None, false);
        sm.add_guarded("closed", "open", "push", "locked==no");
        sm.add_transition("open", "closed", "close");

        let mut ctx = HashMap::new();
        ctx.insert("locked".into(), "yes".into());
        assert!(!sm.handle("push", &ctx)); // guard fails

        ctx.insert("locked".into(), "no".into());
        assert!(sm.handle("push", &ctx)); // guard passes
        assert_eq!(sm.current_state, "open");
    }

    #[test]
    fn test_final_state() {
        let mut sm = StateMachine::new("process", "running");
        sm.add_state("done", None, true);
        sm.add_transition("running", "done", "complete");
        assert!(!sm.is_finished());
        sm.handle("complete", &HashMap::new());
        assert!(sm.is_finished());
    }

    #[test]
    fn test_available_events() {
        let sm = make_traffic_light();
        let events = sm.available_events();
        assert!(events.contains(&"timer"));
    }

    #[test]
    fn test_history() {
        let mut sm = make_traffic_light();
        sm.handle("timer", &HashMap::new());
        sm.handle("timer", &HashMap::new());
        assert_eq!(sm.history.len(), 2);
        assert_eq!(sm.history[0].from, "green");
    }

    #[test]
    fn test_hierarchical_state() {
        let mut sm = StateMachine::new("robot", "idle");
        sm.add_state("working", None, false);
        sm.add_state("lifting", Some("working"), false);
        sm.add_transition("idle", "working", "start");
        sm.add_transition("working", "lifting", "grab");
        sm.handle("start", &HashMap::new());
        sm.handle("grab", &HashMap::new());
        assert_eq!(sm.current_state, "lifting");
        let path = sm.state_path();
        assert_eq!(path, vec!["working", "lifting"]);
    }

    #[test]
    fn test_event_counts() {
        let mut sm = make_traffic_light();
        sm.handle("timer", &HashMap::new());
        sm.handle("timer", &HashMap::new());
        assert_eq!(*sm.event_counts.get("timer").unwrap(), 2);
    }

    #[test]
    fn test_summary() {
        let sm = make_traffic_light();
        let s = sm.summary();
        assert!(s.contains("state=green"));
    }
}
