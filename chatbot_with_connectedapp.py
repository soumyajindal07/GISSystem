from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime
from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import create_agent, AgentState
from typing import Any
from dotenv import load_dotenv
import os

class ChatBot:
    def __init__(self):
      
        load_dotenv()
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        self.memory = InMemorySaver()
        
        self.llm = ChatGroq(model_name="qwen/qwen3-32b",temperature=0, max_tokens= 1200) 
        
        #self.db = SQLDatabase.from_uri("sqlite:///gis_chatbot_final.db")        
         
        
    @before_model
    def trim_messages(self, state: AgentState) -> dict[str, Any] | None:
        """Keep only last few messages for context window."""
        messages = state["messages"]

        if len(messages) <= 3:
            return None  # nothing to trim

        first_msg = messages[0]
        recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
        new_messages = [first_msg] + recent_messages

        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages
            ]
        }
    
    def init_agent(self, istoolAttached: bool = True):
        if istoolAttached:
            self.db = SQLDatabase.from_uri("sqlite:///gis_chatbot_final.db")  
                     
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            self.config = {"configurable": {"thread_id": "user_session_1"}} 
            self.tools = toolkit.get_tools()
            
            self.prompt = """
           You are a GIS operations assistant with access ONLY to the SQLite database
`gis_chatbot_final.db`.

AVAILABLE TABLES (USE ONLY THESE):
- regions(region_id, region_name)
- districts(district_id, district_name, region_id)
- assets(asset_id, asset_type, region_id, district_id, criticality, status, latitude, longitude)
- schools(school_id, school_name, region_id, district_id, level, latitude, longitude)
- incidents(incident_id, asset_id, incident_type, severity, incident_date, district_id, region_id)
- work_orders(work_id, asset_id, work_type, planned_date, district_id, region_id)

====================
CORE RULES
====================
- ALWAYS use the attached SQL tool
- Generate valid SELECT queries only
- NEVER perform INSERT, UPDATE, DELETE, DROP
- If a query fails, fix and retry
- If no data/tool applies, respond EXACTLY:
  "I am a GIS ticketing system, hence cannot respond to this query."

====================
DATA INTEGRITY (STRICT)
====================
- Answers MUST come ONLY from SQL results
- NEVER invent, infer, normalize, or rename any name
- If a value is not returned by SQL, DO NOT mention it
- Return human-readable names only:
  region_name, district_name, school_name
- NEVER return raw latitude/longitude

====================
TIME & INTERPRETATION
====================
- Do NOT mention months or years unless SQL returns them
- If date('now') is used, say "current period" or "last 30 days"
- DO NOT equate incident count with risk
- Use the word "risk" ONLY if SQL calculates:
  severity weighting OR trends OR normalization
- Otherwise describe results as counts or volume

====================
RANKING
====================
- If values are equal, state they are tied
- Do NOT rank tied entities
- Avoid "most/least" unless values differ

====================
EXPLAINABILITY
====================
- Briefly explain what each metric represents
- State limitations if metrics are simple counts

====================
FORMAT
====================
- Output Markdown only
- Use headings and lists
- Use ``` blocks ONLY for SQL
- Use Mermaid pie charts ONLY when visualization adds insight
- No text outside Markdown
    """.format(dialect=self.db.dialect)
        else:
            self.db = SQLDatabase.from_uri("sqlite:///jendela_data.db")
            
            toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
            self.tools = toolkit.get_tools()
            
            self.config = {"configurable": {"thread_id": "user_session_2"}} 
                        
            self.prompt = """
               You are a network coverage and complaint assistant with access to a SQLite database (jendela_data.db).
              Tables:
              - coverage_areas
              - complaints
              - internal_analytics
              - chatbot_feedback

              RULES:
              
              READ:
              - For coverage, availability, complaints, or analytics:  
                You are NOT allowed to answer without calling the database tool first
                • MUST QUERY the database tool
                • Use existing tables only; do not create or infer data
                • When filtering by district or state, always use TRIM and case-insensitive matching.
                • Summarize results clearly
                
                ONLY IF:
              - the database tool was called AND
              - it returned zero rows OR failed,
              respond exactly:
              "I dont have access to this data, hence cannot respond to this query."

              WRITE (Complaint):

              Detect complaint intent
              Collect only missing: state, district, issue_type
              Normalize district aliases → Petaling
              Map issue_type to:
              No Connectivity | Slow Speed | Call Drop | Service Outage | Poor Coverage
              NEVER call tools or DB
              NEVER show SQL, tuples, or payloads

              Generate complaint_id
              Respond ONLY with:
              Title - Below complaint is raised successfully and details with
              Complaint ID, State, District, Issue Type
              Return exactly ONE complaint


              - Normalize locations:
                • "PJ", "Petaling", "Petaling Jaya" → "Petaling"
                • Infer state from district using DB data only

              - Map issue_type to nearest of:
                No Connectivity | Slow Speed | Call Drop | Service Outage | Poor Coverage

              MEMORY:
              - Use memory only to resolve references like “there” or “same area”
              - Never override database results

              SAFETY:
              - Never fabricate data

              FORMAT:
              - Markdown only
              - Use headings and lists when helpful
              - Use ``` blocks for values
              - Use Mermaid pie charts for charts only
              - No extra text
              - Use user friendly keywords
"""
        
        self.llm.bind_tools(tools = self.tools, tool_choice= 'auto')   
       
        #print(f"memory{list(self.memory.list(self.config))}")
          
        # print(f"actual memory{list(self.memory.list(self.config))}")
        # checkpointer = self.memory
        # checkpoints = list(checkpointer.list(self.config))
        # recent_checkpoints = checkpoints[-4:]
        
        # print(f"memory{list(recent_checkpoints)}")
              
        #self.memory = self.create_inmemory_from_last_messages(self.memory, self.config)
        
        self.agent = create_agent(model=self.llm,
                                  tools = self.tools,
                                  system_prompt = self.prompt,
                                  )
        
        # middleware= [self.trim_messages],
        #                           checkpointer= self.memory
                       
    def qna_chatbot(self, prompt):
        """
        Take user query, generate SQL via LLM, execute in SQLite, return readable answer.
        """
        try: 
            # for step in self.agent.stream({"messages": [{"role": "user", "content": prompt}]}, stream_mode="values", config = self.config):
            #     step["messages"][-1].pretty_print()
                
            result = self.agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config= self.config)
            return result["messages"][-1].content
                
        except Exception as e:
            return f"Error: {str(e)}"

# ===============================
# 6. Test Examples
# ===============================

# if __name__ == "__main__":
#     chat = ChatBot() 
#     chat.init_agent(False)
#     response = chat.qna_chatbot("Show asset coverage by region")
#     print(response)

# if __name__ == "__main__":
#     user_queries = [
#        "Show me last week incidents by region",
#        "Show asset coverage by region",
#       "Show me  weekly incidents by region with plot",#       
#       # "Which districts are hotspots?",
#       # "Planned work orders next week",
#       #"Show me all assets near schools with high incidents last 90 days",
#       #"Which districts are most at risk this month?"
#     ]

#     for q in user_queries:
#         print(f"\nUser: {q}")
#         print("Chatbot Answer:")
#         print(chatbot(q))
