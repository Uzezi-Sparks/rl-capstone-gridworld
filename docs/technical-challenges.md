\# Technical Challenges \& Surprises



\## Bugs Encountered



\- \*\*TabError in gridworld.py\*\*: Mixing tabs and spaces caused repeated

&#x20; IndentationError when editing with Notepad. Fixed by writing a 

&#x20; Python script (fix\_indent.py) to convert all tabs to 4 spaces.



\- \*\*Nested git repository\*\*: A duplicate rl-capstone-gridworld folder 

&#x20; with its own .git was accidentally created inside the project. 

&#x20; Removed with git rm --cached and rmdir.



\- \*\*Wrong class name\*\*: PursuitEvasionEnv was imported as GridWorld 

&#x20; causing ImportError. Diagnosed by scanning for class definitions 

&#x20; programmatically.



\- \*\*Environment missing reset/step\*\*: GridWorld was built for Value 

&#x20; Iteration (DP), not RL training loops. Required manual state 

&#x20; encoding and flat state representation for DQN compatibility.



\- \*\*get\_reward() signature\*\*: Required next\_state argument not obvious 

&#x20; from method name. Discovered through TypeError at runtime.



\## Places I Got Stuck



\- Forward view vs backward view distinction for TD(lambda). 

&#x20; Forward view is theoretical and requires full episode. 

&#x20; Backward view uses eligibility traces and is practical.



\- State representation mismatch between tabular agents 

&#x20; (integer state index) and DQN (flat normalised vector).



\- Notepad unreliable for Python editing — indentation errors 

&#x20; consumed significant debugging time across multiple sessions.



\## Surprises



\- SARSA converged faster than Q-learning early in training 

&#x20; on the reactor environment, despite Q-learning being 

&#x20; theoretically optimal. On-policy safety paid off early.



\- Eligibility traces dramatically simplify credit assignment 

&#x20; in sparse reward environments — the adversary environment 

&#x20; only rewards at goal/capture making traces especially valuable.



\- Pure numpy DQN without PyTorch is viable for small discrete 

&#x20; environments and avoids heavy dependencies.



\## What to Avoid



\- Never use Notepad for Python files. Use VS Code.

\- Always check method signatures before assuming API compatibility.

\- Always verify branch is pushed to remote before assuming 

&#x20; GitHub reflects local state.

