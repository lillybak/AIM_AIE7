// LangGraph Visualization
digraph {
	rankdir=TB
	Start [label=Start shape=oval]
	retrieve [label=Retrieve]
	check_context [label="Check Context"]
	generate [label=Generate]
	fact_check [label="Fact Check"]
	End [label=End shape=oval]
	Start -> retrieve
	retrieve -> check_context
	check_context -> generate
	check_context -> End
	generate -> fact_check
	fact_check -> End [label=pass]
	fact_check -> generate [label=fail]
}
