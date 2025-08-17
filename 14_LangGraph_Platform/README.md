<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording     | 🖼️ Slides        | 👨‍💻 Repo         | 📝 Homework      | 📁 Feedback       |
|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|
| [Session 14: Pre-Work](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150?source=copy_link#247cd547af3d80709683ff380f4cba62)| [Session 14: Deploying Agents to Production](https://www.notion.so/Session-14-Deploying-Agents-to-Production-21dcd547af3d80aba092fcb6c649c150) | [Recording!](https://us02web.zoom.us/rec/share/1YepNUK3kqQnYLY8InMfHv84JeiOMyjMRWOZQ9jfjY86dDPvHMhyoz5Zo04w_tn-.91KwoSPyP6K6u0DC)  (@@5J6DVQ)| [Session 14 Slides](https://www.canva.com/design/DAGvVPg7-mw/IRwoSgDXPEqU-PKeIw8zLg/edit?utm_content=DAGvVPg7-mw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) | You are here! | [Session 14 Assignment: Production Agents](https://forms.gle/nZ7ugE4W9VsC1zXE8) | [AIE7 Feedback 8/7](https://forms.gle/juo8SF5y5XiojFyC9)

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph
  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀
- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned


#### ❓ Question:

What is the purpose of the `chunk_overlap` parameter when using `RecursiveCharacterTextSplitter` to prepare documents for RAG, and what trade-offs arise as you increase or decrease its value?

#### ✅ Answer:
The chunk_overlap parameter in RecursiveCharacterTextSplitter (rag.py 70-71) controls how much text is shared between consecutive chunks when splitting documents for RAG. In this code it is set to 0.  
In rag.py: ` chunk_size=750, chunk_overlap=0, length_function=_tiktoken_len`
the overlap is meansured in tokens (vs characters), e.g.: if chunk_overlap=10, and a token is about 3/4 of 5-6 characters in an "average english word", then overlap of 10 means an overlap of ~40 characters in consecutive chunks (in a character-based measurement). 

##### Trade-offs of Increasing chunk_overlap  
Pros:
* Better context preservation: Important information that spans chunk boundaries is less likely to be lost.
* Improved retrieval accuracy: Related concepts that appear near chunk edges remain together
Reduced information fragmentation: Sentences and paragraphs stay more intact.  

Cons:  
* Increased storage costs: More duplicate content means larger vector databases
* Higher embedding costs: More tokens to embed due to overlapping content
* Potential redundancy: May retrieve the same information multiple times
* Larger memory footprint: More data to process during retrieval

#### Decreasing chunck_overlap
At 0 there is no overlap of content in neighboring chunks.
 
Pros:  
* Lower storage and embedding costs: No duplicate content
* Faster processing: Fewer total chunks to process
* Cleaner retrieval: Less redundant information in results

Cons:
* Context loss: Important information at chunk boundaries may be fragmented
* Reduced retrieval quality: Related concepts might be split across chunks
* Potential missed connections: Semantic relationships across chunk boundaries may be lost

#### ❓ Question:

Your retriever is configured with `search_kwargs={"k": 5}`. How would adjusting `k` likely affect RAGAS metrics such as Context Precision and Context Recall in practice, and why?

#### ✅ Answer:
If k is increased,  the Precision might go down because of additional info that would be less related to the query.
The opposite happens with Context Recall because the higher the k the more additional information we retrieve about the topic.

#### ❓ Question:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

#### ✅ Answer:
The evaluator fits after the agent receives the potential response and judges its helpfulness according to the definition of helpfulness in the prompt. The helpfullness returns Y/N. If it returns a "Y" the graph is terminated. If it returns an N it repeats until the hlpfulness becomes a Y or it hits our limit.
  
In our example code, the helpfulness is not exactly defined. THe prompt just asks for "extremely helpful" judgement, leaving it up to the LLM to put a meaning as to what "helfulness" might mean to the user:  
>> "Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'."  
The result was that the 2 answers were almost identical with the unhelpful answer containing actually an extra piece of info than the "helpful".  


### Task 3 comparing answers:

#### AGENT WITHOUT Helpfulness  

Input
human
What is the MuonClip optimizer, and what paper did it first appear in?
Output
ai
The MuonClip optimizer is an advanced optimization technique designed to improve training stability and efficiency in large-scale machine learning systems, particularly large language models (LLMs) and Mixture-of-Experts models. It builds upon the Muon optimizer by incorporating a novel QK-clip mechanism that adaptively rescales attention head parameters to mitigate numerical instabilities such as exploding attention scores during training.  

MuonClip was first introduced in the paper titled "Kimi K2: Open Agentic Intelligence," published on July 28, 2025. This paper details how MuonClip enabled the stable and efficient training of the Kimi K2 model, <mark>which has 32 billion activated parameters</mark> and was trained on 15.5 trillion tokens <maek>without any training crashes. </mark > 


#### AGENT WITH Helpfulness  
agent
ai
The MuonClip optimizer is an advanced optimization technique designed to improve the stability and efficiency of training large-scale machine learning models, particularly large language models (LLMs) and Mixture-of-Experts models. It builds upon the Muon optimizer by incorporating a novel QK-clip mechanism that adaptively rescales attention head parameters to mitigate numerical instabilities such as exploding attention scores during training. This allows for stable training of extremely large models without crashes.

The MuonClip optimizer first appeared in the paper titled "Kimi K2: Open Agentic Intelligence," published on July 28, 2025. This paper introduces Kimi K2, a large language model that was pre-trained on 15.5 trillion tokens <mark>using the MuonClip optimizer</mark>, which played a crucial role in enabling the stable and efficient training of <mark>such a massive model.</mark>

#### ANSWER:   
We notice that the 2 responses are almost identical. If we care to know that the K2 model <mark>has 32 billion activated parameters</mark>, then the UNHELPFUL is the one that provides this response. If we want to learn that the MuonClip has an `optimizer` then the helpful is the one that tells us that.


### 🚧 Advanced Build 🚧 

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.

## 🚧 <mark>THIS BUILD WAS DONE WITHIN MY HOMEWORK # 13. </mark>

### 3 Lessons LEARNED:
1. How to use MCP with Cursor and LangGraph clients
2. How to use LangGraph Platform 
3. The importance of defining clearly in the prompt what we mean with a certain word that the AI's  respond depends upon: e.g. "helpfulness"

### 3 Not learned:
1. How to keep MCP secured
2. How to scale up 
3. If LangGraph is not useful for scaling up, what would be a better platform or method to use.