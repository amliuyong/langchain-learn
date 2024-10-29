# how to create agent with RAG, memory and tools?


## Agent with Mem in sqldb, RAG and tools

```python

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json

# SQLAlchemy setup
Base = declarative_base()

class Conversation(Base):
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(50), unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = 'messages'
    
    id = Column(Integer, primary_key=True)
    conversation_id = Column(String(50), ForeignKey('conversations.conversation_id'))
    speaker = Column(String(10))  # 'human' or 'ai'
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    conversation = relationship("Conversation", back_populates="messages")

class DatabaseManager:
    def __init__(self, database_url="sqlite:///conversations.db"):
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def create_conversation(self, conversation_id):
        session = self.Session()
        try:
            conversation = Conversation(conversation_id=conversation_id)
            session.add(conversation)
            session.commit()
            return conversation
        finally:
            session.close()
    
    def add_message(self, conversation_id, speaker, content):
        session = self.Session()
        try:
            message = Message(
                conversation_id=conversation_id,
                speaker=speaker,
                content=content
            )
            session.add(message)
            session.commit()
        finally:
            session.close()
    
    def get_conversation_history(self, conversation_id):
        session = self.Session()
        try:
            messages = session.query(Message)\
                .filter(Message.conversation_id == conversation_id)\
                .order_by(Message.created_at).all()
            return [(msg.speaker, msg.content) for msg in messages]
        finally:
            session.close()
    
    def delete_conversation(self, conversation_id):
        session = self.Session()
        try:
            conversation = session.query(Conversation)\
                .filter(Conversation.conversation_id == conversation_id).first()
            if conversation:
                session.delete(conversation)
                session.commit()
        finally:
            session.close()

class ConversationalAgent:
    def __init__(self, db_manager, conversation_id, documents=None):
        self.db_manager = db_manager
        self.conversation_id = conversation_id
        
        # Create vector store
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        else:
            self.vectorstore = Chroma(embedding_function=embeddings)
        
        # Initialize tools
        self.tools = [
            setup_retriever_tool(self.vectorstore),
            weather_tool,
            search_tool
        ]
        
        # Initialize memory and load existing conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.load_memory_from_db()
        
        # Create agent
        self.agent = create_openai_tools_agent(llm, self.tools, prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def load_memory_from_db(self):
        """Load conversation history from database"""
        history = self.db_manager.get_conversation_history(self.conversation_id)
        for speaker, content in history:
            if speaker.lower() == "human":
                self.memory.chat_memory.add_message(HumanMessage(content=content))
            elif speaker.lower() == "ai":
                self.memory.chat_memory.add_message(AIMessage(content=content))
    
    def save_message_to_db(self, speaker, content):
        """Save a message to the database"""
        self.db_manager.add_message(self.conversation_id, speaker, content)
    
    def chat(self, message):
        """Send a message to the agent and get response"""
        # Save human message to DB
        self.save_message_to_db("human", message)
        
        # Get agent response
        response = self.agent_executor.invoke({
            "input": message
        })
        
        # Save AI response to DB
        self.save_message_to_db("ai", response["output"])
        
        return response["output"], response["intermediate_steps"]
    
    def get_conversation_history(self):
        """Get conversation history from database"""
        return self.db_manager.get_conversation_history(self.conversation_id)

# Initialize embeddings and LLM (same as before)
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")

# Create tools (same as before)
def get_current_weather(location: str) -> str:
    return f"The weather in {location} is sunny and 22°C"

def search_web(query: str) -> str:
    return f"Search results for: {query}"

weather_tool = Tool(
    name="get_current_weather",
    description="Get the current weather for a given location",
    func=get_current_weather
)

search_tool = Tool(
    name="search_web",
    description="Search the web for current information",
    func=search_web
)

# Setup retriever tool (same as before)
def setup_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    return create_retriever_tool(
        retriever,
        name="knowledge_base",
        description="Searches and returns information from the knowledge base."
    )

# Create agent prompt (same as before)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to various tools and a knowledge base. 
    Use the tools available to you to provide accurate and helpful responses.
    Always cite your sources when using information from the knowledge base."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Example usage
def main():
    # Initialize database manager
    db_manager = DatabaseManager("sqlite:///conversations.db")
    
    # Create a new conversation
    conversation_id = "user123_session1"
    db_manager.create_conversation(conversation_id)
    
    # Create agent with database support
    agent = ConversationalAgent(db_manager, conversation_id)
    
    # Example conversation
    response, steps = agent.chat("What's the weather like in Paris?")
    print("Response:", response)
    
    # Get conversation history from database
    history = agent.get_conversation_history()
    print("\nConversation History:")
    for speaker, content in history:
        print(f"{speaker}: {content}")
    
    # Continue conversation
    response, steps = agent.chat("Tell me more about the climate there")
    print("\nResponse:", response)
    
    # Get updated history
    history = agent.get_conversation_history()
    print("\nUpdated Conversation History:")
    for speaker, content in history:
        print(f"{speaker}: {content}")

if __name__ == "__main__":
    main()
```


## Agent with Mem, RAG and tools

```python

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools.retriever import create_retriever_tool
from langchain_core.messages import HumanMessage, AIMessage
import datetime
import requests

# Initialize embeddings and LLM
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model="gpt-4-turbo-preview")

# Custom tools (same as before)
def get_current_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny and 22°C"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for: {query}"

# Create tools
weather_tool = Tool(
    name="get_current_weather",
    description="Get the current weather for a given location",
    func=get_current_weather
)

search_tool = Tool(
    name="search_web",
    description="Search the web for current information",
    func=search_web
)

# Create retriever tool (same as before)
def setup_retriever_tool(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    return create_retriever_tool(
        retriever,
        name="knowledge_base",
        description="Searches and returns information from the knowledge base."
    )

# Create agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to various tools and a knowledge base. 
    Use the tools available to you to provide accurate and helpful responses.
    Always cite your sources when using information from the knowledge base.
    
    Available tools:
    - knowledge_base: Search the knowledge base for relevant information
    - get_current_weather: Get current weather for a location
    - search_web: Search the web for current information
    
    Previous conversation context is provided to help you maintain continuity."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

class ConversationalAgent:
    def __init__(self, documents=None):
        # Create vector store
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            self.vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        else:
            # Initialize empty vector store if no documents provided
            self.vectorstore = Chroma(embedding_function=embeddings)
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Setup tools
        self.tools = [
            setup_retriever_tool(self.vectorstore),
            weather_tool,
            search_tool
        ]
        
        # Create agent
        self.agent = create_openai_tools_agent(llm, self.tools, prompt)
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def load_memory(self, conversation_history):
        """
        Load conversation history into memory.
        conversation_history should be a list of tuples (speaker, message)
        where speaker is either "human" or "ai"
        """
        for speaker, message in conversation_history:
            if speaker.lower() == "human":
                self.memory.chat_memory.add_message(HumanMessage(content=message))
            elif speaker.lower() == "ai":
                self.memory.chat_memory.add_message(AIMessage(content=message))
    
    def get_memory(self):
        """Return current conversation history"""
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear conversation history"""
        self.memory.clear()
    
    def chat(self, message):
        """Send a message to the agent and get response"""
        response = self.agent_executor.invoke({
            "input": message
        })
        return response["output"], response["intermediate_steps"]

# Example usage
def main():
    # Sample documents (replace with your actual documents)
    documents = [
        # Add your documents here
    ]
    
    # Create agent
    agent = ConversationalAgent(documents)
    
    # Example of loading previous conversation history
    previous_conversation = [
        ("human", "What's the weather like in Paris?"),
        ("ai", "The weather in Paris is sunny and 22°C"),
        ("human", "What about the historical climate data?"),
        ("ai", "Based on the knowledge base, Paris has experienced significant climate changes...")
    ]
    
    # Load the conversation history
    agent.load_memory(previous_conversation)
    
    # Continue the conversation
    response, steps = agent.chat("Given our previous discussion about Paris, what climate trends do you notice?")
    print("\nResponse:", response)
    print("\nIntermediate steps:", steps)
    
    # Get current memory state
    print("\nCurrent conversation history:")
    for message in agent.get_memory():
        print(f"- {message.type}: {message.content}")
    
    # Clear memory if needed
    # agent.clear_memory()

if __name__ == "__main__":
    main()
```

## EnhancedRAGAgent Class

```python

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor
from langchain.tools.retriever import create_retriever_tool
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from typing import List, Dict
import logging

class EnhancedRAGAgent:
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125", temperature: float = 0):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature
        )
        self.embeddings = OpenAIEmbeddings()
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_vectorstore(self, texts: List[str]) -> Chroma:
        """Create and populate vector store with chunked texts."""
        # Enhanced text splitting with better chunk management
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.create_documents(texts)
        self.logger.info(f"Created {len(chunks)} chunks from source documents")
        
        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )

    def create_tools(self, vectorstore: Chroma) -> List[Dict]:
        """Create tools including enhanced retriever and additional utilities."""
        # Create primary retriever tool with metadata
        retriever = vectorstore.as_retriever(
            search_type="similarity_threshold",
            search_kwargs={
                "k": 5,
                "score_threshold": 0.7,
                "fetch_k": 10
            }
        )
        
        main_retriever_tool = create_retriever_tool(
            retriever,
            name="search_knowledge_base",
            description="Searches the knowledge base for detailed information. Use this for in-depth queries about specific topics."
        )
        
        # Create quick search tool for simple queries
        quick_retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 1}
        )
        
        quick_search_tool = create_retriever_tool(
            quick_retriever,
            name="quick_search",
            description="Quickly checks the knowledge base for simple facts. Use this for straightforward factual queries."
        )
        
        return [main_retriever_tool, quick_search_tool]

    def create_prompt(self) -> ChatPromptTemplate:
        """Create an enhanced prompt template with better tool usage guidance."""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable AI assistant with access to a comprehensive knowledge base.

Available tools:
- search_knowledge_base: Use for detailed research and complex queries
- quick_search: Use for simple factual lookups

Guidelines:
1. For complex queries, use search_knowledge_base and consider multiple relevant pieces of information
2. For simple facts, use quick_search to get immediate answers
3. If uncertain, use search_knowledge_base for thoroughness
4. Combine information from multiple searches when necessary
5. Always cite your sources if they're mentioned in the retrieved content

Think step-by-step:
1. Analyze the query type (complex/simple)
2. Choose appropriate search tool
3. Consider if multiple searches are needed
4. Synthesize information clearly"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def initialize_agent(self, tools) -> AgentExecutor:
        """Initialize the agent with enhanced configuration."""
        prompt = self.create_prompt()
        
        agent = OpenAIFunctionsAgent(
            llm=self.llm,
            tools=tools,
            prompt=prompt
        )
        
        return AgentExecutor(
            agent=agent,
            tools=tools,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )

    def create_agent(self, texts: List[str]) -> AgentExecutor:
        """Create a fully configured agent from provided texts."""
        vectorstore = self.create_vectorstore(texts)
        tools = self.create_tools(vectorstore)
        return self.initialize_agent(tools)

    def query_with_metrics(self, agent: AgentExecutor, query: str) -> Dict:
        """Execute query and return response with usage metrics."""
        with get_openai_callback() as cb:
            try:
                response = agent.invoke({"input": query})
                metrics = {
                    "total_tokens": cb.total_tokens,
                    "prompt_tokens": cb.prompt_tokens,
                    "completion_tokens": cb.completion_tokens,
                    "total_cost": cb.total_cost,
                    "successful": True
                }
            except Exception as e:
                self.logger.error(f"Error during query execution: {str(e)}")
                response = {"error": str(e)}
                metrics = {
                    "total_tokens": cb.total_tokens,
                    "successful": False,
                    "error": str(e)
                }

        return {
            "response": response,
            "metrics": metrics
        }

# Example usage
if __name__ == "__main__":
    # Initialize the enhanced agent
    rag_agent = EnhancedRAGAgent()
    
    # Example knowledge base texts
    texts = [
        "Quantum computing leverages quantum mechanical phenomena...",
        "Machine learning is a subset of artificial intelligence...",
        # Add more texts as needed
    ]
    
    # Create the agent
    agent = rag_agent.create_agent(texts)
    
    # Example queries demonstrating different use cases
    queries = [
        "What is quantum computing?",  # Simple query - should use quick_search
        "Compare quantum computing with classical computing and explain the key differences",  # Complex query - should use search_knowledge_base
        "What are the main applications of machine learning?",  # Moderate complexity - agent decides tool
    ]
    
    # Run queries and display results with metrics
    for query in queries:
        print(f"\nQuery: {query}")
        result = rag_agent.query_with_metrics(agent, query)
        print(f"Response: {result['response']}")
        print(f"Metrics: {result['metrics']}")
 ```

## use enhanced_query

```python 

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_core.messages import AIMessage, HumanMessage
import datetime
import requests

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-4")

# Set up the vector store for RAG
def setup_rag():
    # Load and process documents
    loader = TextLoader("path_to_your_documents.txt")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever

# Custom tools
def get_current_time():
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def search_weather(location: str):
    """Get weather information for a location."""
    # Replace with actual API call
    return f"Weather information for {location} (mock data): Sunny, 22°C"

# Create tools list
tools = [
    Tool(
        name="Current Time",
        func=get_current_time,
        description="Useful for getting the current date and time"
    ),
    Tool(
        name="Weather Search",
        func=search_weather,
        description="Useful for getting weather information for a location"
    )
]

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to several tools and a knowledge base.
    Use the tools when needed, and reference the knowledge base for contextual information.
    Always provide clear, concise responses."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Initialize retriever
retriever = setup_rag()

# Create the agent
agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Function to combine RAG with agent response
async def process_query(query: str):
    # First, get relevant documents from RAG
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in docs])
    
    # Combine the context with the query
    enhanced_query = f"""Context from knowledge base:
    {context}
    
    User query: {query}
    
    Please provide a response based on both the context and your tools."""
    
    # Run the agent
    response = await agent_executor.ainvoke({"input": enhanced_query})
    return response

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Example queries
        queries = [
            "What's the weather like in London and what time is it?",
            "Can you find information about machine learning in our documents?",
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            response = await process_query(query)
            print(f"Response: {response}")
    
    asyncio.run(main())


```


## use retriever as a tool

```python 

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from typing import List

# Initialize the LLM
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")

# Set up the vector store and retriever
def setup_retriever():
    loader = TextLoader("your_knowledge_base.txt")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

# Custom tool functions
def search_weather(location: str) -> str:
    """Search for current weather in a given location."""
    return f"Current weather in {location}: 22°C, Partly Cloudy"

def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    # Get relevant documents
    docs = retriever.get_relevant_documents(query)
    
    # Format the results
    if not docs:
        return "No relevant information found in the knowledge base."
    
    results = []
    for i, doc in enumerate(docs, 1):
        results.append(f"Reference {i}:\n{doc.page_content}")
    
    return "\n\n".join(results)

def search_news(query: str) -> List[str]:
    """Search for recent news articles based on a query."""
    return [f"Recent news about {query}: Sample headline"]

# Initialize the retriever
retriever = setup_retriever()

# Define tools including the retriever as a tool
tools = [
    Tool(
        name="KnowledgeBase",
        func=search_knowledge_base,
        description="Useful for searching information in the knowledge base. Use this for questions about stored knowledge or historical information."
    ),
    Tool(
        name="Weather",
        func=search_weather,
        description="Useful for getting weather information for a location"
    ),
    Tool(
        name="News",
        func=search_news,
        description="Useful for searching recent news articles"
    )
]

# Set up memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant with access to various tools:
    - KnowledgeBase: Search the knowledge base for historical or stored information
    - Weather: Get current weather for a location
    - News: Search recent news articles
    
    When asked about any historical information or stored knowledge, always use the KnowledgeBase tool first.
    For current weather, use the Weather tool.
    For recent events, use the News tool.
    
    Think step by step and explain which tool you're using and why."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create and initialize the agent
agent = create_openai_tools_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# Example usage
if __name__ == "__main__":
    # Example queries
    queries = [
        "What's the weather like in London?",
        "What do we know about quantum computing?",
        "Tell me about recent AI developments and compare them with our historical knowledge."
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        response = agent_executor.invoke({"input": query})
        print(f"Response: {response}")

```

## More examples

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document
from typing import List, Dict

# Initialize common components
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings()

# Method 1: Using RetrievalQA Chain as a Tool
def setup_retrieval_qa_agent():
    # Setup vector store
    texts = ["Your knowledge base content here"]
    docs = [Document(page_content=t) for t in texts]
    vectorstore = Chroma.from_documents(docs, embeddings)
    
    # Create RetrievalQA chain
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Create tool from the chain
    tools = [
        Tool(
            name="KnowledgeBase",
            func=retrieval_qa.invoke,
            description="Use this tool for querying the knowledge base. Input should be a question."
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant with access to a knowledge base. Use the KnowledgeBase tool to answer questions."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

# Method 2: Using OpenAI Functions Agent with Retriever Tool
def setup_functions_agent():
    vectorstore = Chroma.from_documents(
        [Document(page_content="Your knowledge base content here")],
        embeddings
    )
    
    # Create retriever tool using LangChain's utility
    retriever_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        name="search_knowledge_base",
        description="Searches the knowledge base for relevant information"
    )
    
    tools = [retriever_tool]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Use the search_knowledge_base tool to find relevant information."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(agent=agent, tools=tools)

# Method 3: Using Conversational Retrieval Chain with Custom Agent
from langchain.chains import ConversationalRetrievalChain

def setup_conversational_agent():
    vectorstore = Chroma.from_documents(
        [Document(page_content="Your knowledge base content here")],
        embeddings
    )
    
    # Create conversational retrieval chain
    retrieval_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True
    )
    
    def retrieve_and_format(query: str) -> str:
        result = retrieval_chain({"question": query, "chat_history": []})
        return f"Answer: {result['answer']}\nSources: {[doc.page_content[:100] + '...' for doc in result['source_documents']]}"
    
    tools = [
        Tool(
            name="ConversationalKB",
            func=retrieve_and_format,
            description="Use this for queries that require context-aware searching of the knowledge base"
        )
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant with access to a conversational knowledge base."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)

# Method 4: Using Parent-Child Agent Architecture
def setup_parent_child_agent():
    # Setup vector store
    vectorstore = Chroma.from_documents(
        [Document(page_content="Your knowledge base content here")],
        embeddings
    )
    
    # Create child agent for knowledge base queries
    kb_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a specialized agent for querying the knowledge base. Be precise and detailed."),
        ("user", "{input}")
    ])
    
    def query_kb(query: str) -> str:
        docs = vectorstore.similarity_search(query)
        context = "\n".join(doc.page_content for doc in docs)
        return llm.invoke(kb_prompt.format_messages(input=f"Context: {context}\nQuestion: {query}")).content
    
    # Create parent agent with access to child agent
    tools = [
        Tool(
            name="KnowledgeExpert",
            func=query_kb,
            description="Use this when you need detailed information from the knowledge base"
        )
    ]
    
    parent_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a parent agent that can delegate knowledge base queries to a specialized expert agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    parent_agent = create_openai_tools_agent(llm, tools, parent_prompt)
    return AgentExecutor(agent=parent_agent, tools=tools)

# Example usage
if __name__ == "__main__":
    # Initialize different agent types
    retrieval_qa_agent = setup_retrieval_qa_agent()
    functions_agent = setup_functions_agent()
    conversational_agent = setup_conversational_agent()
    parent_child_agent = setup_parent_child_agent()
    
    # Example query
    query = "What do we know about quantum computing?"
    
    # Test different approaches
    print("\nRetrievalQA Agent Response:")
    print(retrieval_qa_agent.invoke({"input": query}))
    
    print("\nFunctions Agent Response:")
    print(functions_agent.invoke({"input": query}))
    
    print("\nConversational Agent Response:")
    print(conversational_agent.invoke({"input": query}))
    
    print("\nParent-Child Agent Response:")
    print(parent_child_agent.invoke({"input": query}))

```

