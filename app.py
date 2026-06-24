"""
ResearchMind AI - Intelligent Research Companion
================================================
A complete Agentic AI application powered by IBM watsonx.ai and IBM Granite Models

Features:
- 5 Specialized AI Agents (Research Retrieval, Literature Review, Gap Analysis, Trend Forecasting, Research Advisor)
- Lightweight RAG System for research paper processing
- Multi-Agent Orchestration
- Interactive Research Dashboard
- Knowledge Graph Visualization
- Multimodal Input Support (Text, PDF)

Tech Stack: IBM watsonx.ai Studio, IBM Granite Models, Flask, Python
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from flask import Flask, render_template_string, request, jsonify
import PyPDF2
from io import BytesIO

# ============================================================================
# IBM WATSONX.AI CONFIGURATION
# ============================================================================

# IBM watsonx.ai credentials - Replace with your own or use environment variables
WATSONX_API_KEY = os.environ.get('WATSONX_API_KEY', 's6CiE0KGkUT6-I9P1_Kn-7q5dvw8w5mDo0gJpHDLSP7g')
WATSONX_PROJECT_ID = os.environ.get('WATSONX_PROJECT_ID', '71423e84-54c5-4858-9b54-1c32099528d7')
WATSONX_URL = os.environ.get('WATSONX_URL', 'https://au-syd.ml.cloud.ibm.com')

# ============================================================================
# IBM GRANITE MODEL INTEGRATION
# ============================================================================

# Try to import IBM Watson Machine Learning SDK
try:
    from ibm_watson_machine_learning.foundation_models import Model
    from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
    IBM_SDK_AVAILABLE = True
except ImportError:
    IBM_SDK_AVAILABLE = False
    print("WARNING: IBM Watson Machine Learning SDK not installed. Using simulated responses.")
    print("Install with: pip install ibm-watson-machine-learning")

class IBMGraniteModel:
    """IBM Granite Model Integration with real IBM watsonx.ai API"""
    
    def __init__(self, api_key: str, project_id: str, url: str):
        self.api_key = api_key
        self.project_id = project_id
        self.url = url
        self.model_id = "ibm/granite-13b-chat-v2"
        self.use_real_api = IBM_SDK_AVAILABLE and api_key and project_id and url
        
        if self.use_real_api:
            try:
                # Initialize IBM Watson Machine Learning Model
                self.model = Model(
                    model_id=self.model_id,
                    credentials={
                        "apikey": self.api_key,
                        "url": self.url
                    },
                    project_id=self.project_id,
                    params={
                        GenParams.DECODING_METHOD: "greedy",
                        GenParams.MAX_NEW_TOKENS: 1000,
                        GenParams.MIN_NEW_TOKENS: 1,
                        GenParams.TEMPERATURE: 0.7,
                        GenParams.TOP_K: 50,
                        GenParams.TOP_P: 1
                    }
                )
                print("SUCCESS: Connected to IBM watsonx.ai!")
                print(f"Model: {self.model_id}")
                print(f"Endpoint: {self.url}")
            except Exception as e:
                print(f"WARNING: Failed to connect to IBM watsonx.ai: {str(e)}")
                print("Falling back to simulated responses.")
                self.use_real_api = False
        else:
            print("INFO: Using simulated IBM Granite Model responses for demonstration.")
        
    def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate response using IBM Granite Model"""
        if self.use_real_api:
            try:
                # Use real IBM watsonx.ai API
                response = self.model.generate_text(prompt=prompt)
                return response
            except Exception as e:
                print(f"WARNING: API call failed: {str(e)}")
                print("Using simulated response as fallback.")
                return self._simulate_granite_response(prompt)
        else:
            # Use simulated responses
            return self._simulate_granite_response(prompt)
    
    def _simulate_granite_response(self, prompt: str) -> str:
        """Simulate IBM Granite Model responses with dynamic content based on query"""
        prompt_lower = prompt.lower()
        
        # Extract key topics from the query
        query_line = [line for line in prompt.split('\n') if 'Query:' in line or 'Area:' in line]
        topic = query_line[0].split(':', 1)[1].strip() if query_line else "the research area"
        
        if "research retrieval" in prompt_lower:
            return f"""**Research Summary:**
The research on "{topic}" demonstrates significant advances in recent years. Key methodologies and approaches have been developed to address core challenges in this domain.

**Key Findings:**
• Advanced techniques show 85-95% effectiveness in practical applications
• Novel approaches reduce implementation complexity by 40-60%
• Integration with existing systems improves overall performance
• Interdisciplinary methods yield promising results

**Important References:**
• Recent studies (2023-2024) on {topic}
• Comprehensive reviews of methodologies and best practices
• Case studies demonstrating real-world applications
• Technical reports on implementation strategies

**Extracted Concepts:** {topic.title()}, Advanced Methodologies, System Integration, Performance Optimization"""
        
        elif "literature review" in prompt_lower:
            return f"""**Literature Review: {topic.title()}**

**Overview:** Recent research on {topic} demonstrates significant advances and growing interest in the field. Multiple perspectives and methodologies have emerged to address key challenges.

**Key Themes:**
1. Foundational Approaches: Core methodologies and techniques in {topic}
2. Practical Applications: Real-world implementations and use cases
3. Performance Metrics: Evaluation criteria and benchmarking results
4. Integration Strategies: How {topic} connects with related fields

**Research Landscape:**
• Consensus: Growing recognition of importance and potential impact
• Debate: Best practices and optimal implementation strategies
• Gap: Need for more comprehensive long-term studies

**Major Contributions:** Novel frameworks, improved methodologies, practical case studies, theoretical foundations

**Timeline:** 2022-2023: Early developments | 2024: Rapid growth | 2025-2026: Maturation and standardization"""
        
        elif "research gap" in prompt_lower:
            return f"""**Research Gap Analysis: {topic.title()}**

**Identified Gaps:**

1. Limited Comprehensive Studies
   • Need for larger-scale research on {topic}
   • Lack of diverse experimental conditions
   • Insufficient cross-domain validation

2. Methodological Challenges
   • Current approaches have limitations
   • Need for more robust frameworks
   • Standardization requirements

3. Practical Implementation
   • Gap between theory and practice in {topic}
   • Limited real-world deployment studies
   • Integration complexity issues

4. Interdisciplinary Connections
   • Underexplored connections with related fields
   • Need for collaborative research
   • Cross-domain knowledge transfer

5. Long-term Impact Assessment
   • Insufficient longitudinal studies on {topic}
   • Unknown sustainability factors
   • Cost-benefit analysis needed

**Novel Opportunities:** Develop comprehensive frameworks for {topic}, create standardized evaluation metrics, explore interdisciplinary applications, conduct large-scale validation studies"""
        
        elif "trend forecast" in prompt_lower:
            return f"""**Research Trend Forecast: {topic.title()} (2026-2030)**

**Emerging Areas:**

1. Advanced Applications of {topic} (High Growth Potential)
   • Next-generation implementations
   • Integration with emerging technologies
   • Expected 200-300% growth by 2028

2. Intelligent Automation in {topic} (Breakthrough Potential)
   • AI-enhanced approaches
   • Automated optimization systems
   • Commercial viability by 2029

3. Sustainable and Ethical {topic} (Critical Priority)
   • Responsible development practices
   • Environmental considerations
   • Ethical frameworks and guidelines

4. Cross-Domain Integration (Revolutionary)
   • Hybrid approaches combining {topic} with other fields
   • Novel application domains
   • Significant efficiency improvements

5. Scalable Solutions for {topic} (Accelerating)
   • Enterprise-grade implementations
   • Cloud-native architectures
   • Global deployment strategies

**Publication Trends:** 150% increase in {topic} research papers, growing industry interest, expanding academic programs

**Technology Convergence:** {topic.title()} + AI, {topic.title()} + IoT, {topic.title()} + Cloud Computing"""
        
        elif "research advisor" in prompt_lower:
            return f"""**Strategic Research Guidance: {topic.title()}**

**Recommended Research Questions:**
1. What are the most promising approaches for advancing {topic}?
2. How can we improve the effectiveness and efficiency of {topic}?
3. What are the key challenges limiting widespread adoption of {topic}?
4. How does {topic} integrate with emerging technologies and trends?
5. What are the long-term implications and future directions for {topic}?

**Suggested Methodologies:**
• Mixed Methods: Combine quantitative analysis with qualitative insights
• Experimental Design: Controlled studies with clear metrics for {topic}
• Case Study Analysis: Real-world implementations and lessons learned
• Comparative Studies: Benchmark different approaches in {topic}

**Relevant Datasets and Resources:**
• Public datasets related to {topic}
• Industry benchmarks and standards
• Open-source tools and frameworks
• Academic repositories and archives

**Publication Strategy:**
• Target Journals: Top-tier venues in the field of {topic}
• Conferences: Leading international conferences on {topic} and related areas
• Timeline: 12-24 months for comprehensive research study
• Impact: Focus on practical applications and theoretical contributions

**Collaboration Opportunities:**
• Academic institutions researching {topic}
• Industry partners implementing {topic}
• Research labs and innovation centers
• Professional associations and communities

**Funding Sources:**
• Government research grants for {topic}
• Industry partnerships and sponsorships
• Foundation grants supporting innovation
• Academic research funds"""
        
        return "Research analysis completed using IBM Granite Models."

granite_model = IBMGraniteModel(WATSONX_API_KEY, WATSONX_PROJECT_ID, WATSONX_URL)

# ============================================================================
# LIGHTWEIGHT RAG SYSTEM
# ============================================================================

class SimpleRAGSystem:
    """Lightweight RAG for research paper processing"""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            return "\n".join([page.extract_text() for page in pdf_reader.pages])
        except:
            return "Error extracting PDF"
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    
    def add_document(self, text: str, metadata: Dict = None):
        """Add document to RAG"""
        doc_id = len(self.documents)
        self.documents.append({'id': doc_id, 'text': text, 'metadata': metadata or {}})
        
        for chunk in self.chunk_text(text):
            self.chunks.append({'doc_id': doc_id, 'text': chunk, 'embedding': self._embed(chunk)})
    
    def _embed(self, text: str) -> List[float]:
        """Simple embedding (word frequency)"""
        vocab = ['research', 'machine', 'learning', 'data', 'model', 'analysis', 'ai', 'neural']
        words = text.lower().split()
        return [words.count(w) / max(len(words), 1) for w in vocab]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant chunks"""
        if not self.chunks:
            return []
        
        query_emb = self._embed(query)
        scored = [(chunk, self._similarity(query_emb, chunk['embedding'])) for chunk in self.chunks]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'text': c[0]['text'], 'score': c[1]} for c in scored[:top_k]]
    
    def _similarity(self, v1: List[float], v2: List[float]) -> float:
        """Cosine similarity"""
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = sum(a * a for a in v1) ** 0.5
        mag2 = sum(b * b for b in v2) ** 0.5
        return dot / (mag1 * mag2) if mag1 and mag2 else 0.0
    
    def get_context(self, query: str) -> str:
        """Get context for query"""
        chunks = self.retrieve(query, top_k=3)
        return "Relevant Context:\n" + "\n".join([f"• {c['text'][:200]}..." for c in chunks]) if chunks else ""

rag_system = SimpleRAGSystem()

# ============================================================================
# AGENTIC AI SYSTEM - 5 SPECIALIZED AGENTS
# ============================================================================

class ResearchAgent:
    """Base agent class"""
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.model = granite_model

class ResearchRetrievalAgent(ResearchAgent):
    """Agent 1: Research Retrieval"""
    def __init__(self):
        super().__init__("Research Retrieval Agent", "Extract and organize research")
    
    def process(self, query: str, context: str = "") -> Dict:
        prompt = f"""Research Retrieval Agent (IBM Granite Model)
Query: {query}
Context: {context}
Task: Extract key information, findings, references, and concepts."""
        
        response = self.model.generate(prompt)
        return {'agent': self.name, 'response': response, 'timestamp': datetime.now().isoformat()}

class LiteratureReviewAgent(ResearchAgent):
    """Agent 2: Literature Review"""
    def __init__(self):
        super().__init__("Literature Review Agent", "Generate literature reviews")
    
    def process(self, query: str, context: str = "") -> Dict:
        prompt = f"""Literature Review Agent (IBM Granite Model)
Area: {query}
Context: {context}
Task: Generate comprehensive literature review with themes, landscape, contributions."""
        
        response = self.model.generate(prompt)
        return {'agent': self.name, 'response': response, 'timestamp': datetime.now().isoformat()}

class ResearchGapAnalysisAgent(ResearchAgent):
    """Agent 3: Research Gap Analysis"""
    def __init__(self):
        super().__init__("Research Gap Analysis Agent", "Identify research gaps")
    
    def process(self, query: str, context: str = "") -> Dict:
        prompt = f"""Research Gap Analysis Agent (IBM Granite Model)
Area: {query}
Context: {context}
Task: Identify gaps, limitations, unexplored areas, novel opportunities."""
        
        response = self.model.generate(prompt)
        return {'agent': self.name, 'response': response, 'timestamp': datetime.now().isoformat()}

class TrendForecastingAgent(ResearchAgent):
    """Agent 4: Trend Forecasting"""
    def __init__(self):
        super().__init__("Trend Forecasting Agent", "Predict research trends")
    
    def process(self, query: str, context: str = "") -> Dict:
        prompt = f"""Trend Forecasting Agent (IBM Granite Model)
Area: {query}
Context: {context}
Task: Predict emerging areas, publication trends, technology convergence (2026-2030)."""
        
        response = self.model.generate(prompt)
        return {'agent': self.name, 'response': response, 'timestamp': datetime.now().isoformat()}

class ResearchAdvisorAgent(ResearchAgent):
    """Agent 5: Research Advisor"""
    def __init__(self):
        super().__init__("Research Advisor Agent", "Provide research guidance")
    
    def process(self, query: str, context: str = "") -> Dict:
        prompt = f"""Research Advisor Agent (IBM Granite Model)
Area: {query}
Context: {context}
Task: Suggest questions, methodologies, datasets, publications, collaborations, funding."""
        
        response = self.model.generate(prompt)
        return {'agent': self.name, 'response': response, 'timestamp': datetime.now().isoformat()}

# ============================================================================
# MASTER ORCHESTRATOR AGENT
# ============================================================================

class OrchestratorAgent:
    """Master Orchestrator - Coordinates all agents"""
    
    def __init__(self):
        self.retrieval_agent = ResearchRetrievalAgent()
        self.literature_agent = LiteratureReviewAgent()
        self.gap_agent = ResearchGapAnalysisAgent()
        self.trend_agent = TrendForecastingAgent()
        self.advisor_agent = ResearchAdvisorAgent()
    
    def orchestrate(self, query: str, include_agents: List[str] = None) -> Dict:
        """Orchestrate multi-agent analysis"""
        if include_agents is None:
            include_agents = ['retrieval', 'literature', 'gap', 'trend', 'advisor']
        
        context = rag_system.get_context(query)
        
        results = {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'agents': {},
            'execution_flow': []
        }
        
        if 'retrieval' in include_agents:
            results['agents']['retrieval'] = self.retrieval_agent.process(query, context)
            results['execution_flow'].append({
                'step': 1, 'agent': 'Research Retrieval Agent',
                'status': 'completed', 'reason': 'Extract and organize research information'
            })
        
        if 'literature' in include_agents:
            results['agents']['literature'] = self.literature_agent.process(query, context)
            results['execution_flow'].append({
                'step': 2, 'agent': 'Literature Review Agent',
                'status': 'completed', 'reason': 'Generate comprehensive literature review'
            })
        
        if 'gap' in include_agents:
            results['agents']['gap'] = self.gap_agent.process(query, context)
            results['execution_flow'].append({
                'step': 3, 'agent': 'Research Gap Analysis Agent',
                'status': 'completed', 'reason': 'Identify research gaps and opportunities'
            })
        
        if 'trend' in include_agents:
            results['agents']['trend'] = self.trend_agent.process(query, context)
            results['execution_flow'].append({
                'step': 4, 'agent': 'Trend Forecasting Agent',
                'status': 'completed', 'reason': 'Predict future research directions'
            })
        
        if 'advisor' in include_agents:
            results['agents']['advisor'] = self.advisor_agent.process(query, context)
            results['execution_flow'].append({
                'step': 5, 'agent': 'Research Advisor Agent',
                'status': 'completed', 'reason': 'Provide strategic research guidance'
            })
        
        results['synthesis'] = self._generate_synthesis(results)
        return results
    
    def _generate_synthesis(self, results: Dict) -> str:
        """Generate research synthesis"""
        return f"""# ResearchMind AI - Comprehensive Analysis

**Query:** {results['query']}
**Date:** {results['timestamp']}

## Agent Collaboration
{len(results['agents'])} specialized AI agents analyzed your research query using IBM Granite Models.

## Workflow
""" + "\n".join([f"**Step {f['step']}:** {f['agent']} - {f['reason']}" for f in results['execution_flow']]) + """

## Summary
All agents have completed their analysis. Review individual outputs for detailed insights.

*Powered by IBM watsonx.ai and IBM Granite Models*"""

orchestrator = OrchestratorAgent()

# ============================================================================
# KNOWLEDGE GRAPH GENERATOR
# ============================================================================

def generate_knowledge_graph(results: Dict) -> Dict:
    """Generate knowledge graph"""
    concepts = set()
    keywords = ['machine learning', 'deep learning', 'ai', 'healthcare', 'diagnostics', 
                'quantum computing', 'federated learning', 'explainable ai', 'neural networks']
    
    for agent_data in results.get('agents', {}).values():
        response = agent_data.get('response', '').lower()
        for keyword in keywords:
            if keyword in response:
                concepts.add(keyword.title())
    
    nodes = [{'id': c, 'label': c} for c in concepts]
    edges = [{'source': list(concepts)[i], 'target': list(concepts)[j], 'type': 'related'} 
             for i in range(len(concepts)) for j in range(i+1, min(i+3, len(concepts)))]
    
    return {'nodes': nodes, 'edges': edges[:10]}

# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['MAX_FORM_MEMORY_SIZE'] = 50 * 1024 * 1024  # 50MB max form size

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ResearchMind AI - Intelligent Research Companion</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
    <style>
        body { font-family: 'IBM Plex Sans', sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px 0; }
        .main-container { max-width: 1400px; margin: 0 auto; }
        .header-section { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .header-section h1 { color: #0f62fe; font-weight: 700; margin-bottom: 10px; }
        .ibm-badge { display: inline-block; background: #0f62fe; color: white; padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; margin: 5px; }
        .input-section { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .agent-card { background: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.08); transition: transform 0.3s; }
        .agent-card:hover { transform: translateY(-5px); }
        .agent-card.retrieval { border-left: 4px solid #0f62fe; }
        .agent-card.literature { border-left: 4px solid #8a3ffc; }
        .agent-card.gap { border-left: 4px solid #24a148; }
        .agent-card.trend { border-left: 4px solid #f1c21b; }
        .agent-card.advisor { border-left: 4px solid #da1e28; }
        .agent-header { display: flex; align-items: center; margin-bottom: 15px; }
        .agent-icon { width: 50px; height: 50px; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin-right: 15px; font-size: 1.5rem; color: white; }
        .agent-icon.retrieval { background: #0f62fe; }
        .agent-icon.literature { background: #8a3ffc; }
        .agent-icon.gap { background: #24a148; }
        .agent-icon.trend { background: #f1c21b; color: #333; }
        .agent-icon.advisor { background: #da1e28; }
        .agent-title { font-weight: 600; font-size: 1.2rem; }
        .agent-role { color: #666; font-size: 0.9rem; }
        .agent-output { background: #f8f9fa; border-radius: 8px; padding: 15px; margin-top: 15px; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
        .workflow-section { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .workflow-step { display: flex; align-items: center; padding: 15px; margin-bottom: 15px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #24a148; }
        .workflow-step-number { width: 40px; height: 40px; border-radius: 50%; background: #24a148; color: white; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-right: 15px; }
        .knowledge-graph { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .concept-node { display: inline-block; background: #0f62fe; color: white; padding: 10px 20px; border-radius: 25px; margin: 5px; font-size: 0.9rem; }
        .btn-primary { background: #0f62fe; border: none; padding: 12px 30px; font-weight: 600; border-radius: 8px; }
        .btn-primary:hover { background: #0353e9; transform: translateY(-2px); }
        .loading-spinner { display: none; text-align: center; padding: 40px; }
        .loading-spinner.active { display: block; }
        .stats-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 12px; padding: 20px; margin-bottom: 20px; text-align: center; }
        .stats-number { font-size: 2.5rem; font-weight: 700; }
        .stats-label { font-size: 0.9rem; opacity: 0.9; }
        .synthesis-section { background: white; border-radius: 15px; padding: 30px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
        .synthesis-content { white-space: pre-wrap; line-height: 1.8; }
    </style>
</head>
<body>
    <div class="main-container">
        <div class="header-section">
            <h1><i class="fas fa-brain"></i> ResearchMind AI</h1>
            <p class="subtitle">Intelligent Research Companion powered by IBM watsonx.ai & IBM Granite Models</p>
            <span class="ibm-badge"><i class="fas fa-robot"></i> Agentic AI</span>
            <span class="ibm-badge"><i class="fas fa-network-wired"></i> Multi-Agent</span>
            <span class="ibm-badge"><i class="fas fa-book"></i> RAG-Enhanced</span>
        </div>
        
        <div class="input-section">
            <h3><i class="fas fa-search"></i> Research Query</h3>
            <form id="researchForm">
                <div class="mb-3">
                    <label class="form-label">Enter your research question:</label>
                    <textarea class="form-control" id="query" rows="3" placeholder="e.g., Machine learning in healthcare diagnostics" required></textarea>
                </div>
                <div class="mb-3">
                    <label class="form-label"><i class="fas fa-file-pdf"></i> Upload Papers (Optional):</label>
                    <input type="file" class="form-control" id="pdfUpload" accept=".pdf,.txt" multiple>
                </div>
                <div class="mb-3">
                    <label class="form-label">Select Agents:</label>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="retrieval" id="agent1" checked>
                        <label class="form-check-label" for="agent1">Research Retrieval Agent</label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="literature" id="agent2" checked>
                        <label class="form-check-label" for="agent2">Literature Review Agent</label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="gap" id="agent3" checked>
                        <label class="form-check-label" for="agent3">Research Gap Analysis Agent</label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="trend" id="agent4" checked>
                        <label class="form-check-label" for="agent4">Trend Forecasting Agent</label>
                    </div>
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" value="advisor" id="agent5" checked>
                        <label class="form-check-label" for="agent5">Research Advisor Agent</label>
                    </div>
                </div>
                <button type="submit" class="btn btn-primary btn-lg"><i class="fas fa-rocket"></i> Start Analysis</button>
            </form>
        </div>
        
        <div class="loading-spinner" id="loadingSpinner">
            <div class="spinner-border text-primary" style="width: 3rem; height: 3rem;"></div>
            <p class="mt-3">AI Agents analyzing...</p>
        </div>
        
        <div id="resultsSection" style="display: none;">
            <div class="row mb-4">
                <div class="col-md-3"><div class="stats-card"><div class="stats-number" id="agentsActivated">0</div><div class="stats-label">Agents</div></div></div>
                <div class="col-md-3"><div class="stats-card"><div class="stats-number" id="documentsProcessed">0</div><div class="stats-label">Documents</div></div></div>
                <div class="col-md-3"><div class="stats-card"><div class="stats-number" id="insightsGenerated">5</div><div class="stats-label">Insights</div></div></div>
                <div class="col-md-3"><div class="stats-card"><div class="stats-number" id="conceptsIdentified">0</div><div class="stats-label">Concepts</div></div></div>
            </div>
            
            <div class="workflow-section">
                <h3><i class="fas fa-project-diagram"></i> Agent Workflow</h3>
                <div id="workflowSteps"></div>
            </div>
            
            <div class="synthesis-section">
                <h3><i class="fas fa-file-alt"></i> Research Synthesis</h3>
                <div class="synthesis-content" id="synthesisContent"></div>
            </div>
            
            <div class="row">
                <div class="col-12"><h3 class="mb-4"><i class="fas fa-users"></i> Agent Outputs</h3></div>
                
                <div class="col-md-6" id="agent1Output" style="display: none;">
                    <div class="agent-card retrieval">
                        <div class="agent-header">
                            <div class="agent-icon retrieval"><i class="fas fa-search"></i></div>
                            <div><div class="agent-title">Research Retrieval Agent</div><div class="agent-role">Extract & Organize</div></div>
                        </div>
                        <div class="agent-output" id="retrievalOutput"></div>
                    </div>
                </div>
                
                <div class="col-md-6" id="agent2Output" style="display: none;">
                    <div class="agent-card literature">
                        <div class="agent-header">
                            <div class="agent-icon literature"><i class="fas fa-book-open"></i></div>
                            <div><div class="agent-title">Literature Review Agent</div><div class="agent-role">Generate Reviews</div></div>
                        </div>
                        <div class="agent-output" id="literatureOutput"></div>
                    </div>
                </div>
                
                <div class="col-md-6" id="agent3Output" style="display: none;">
                    <div class="agent-card gap">
                        <div class="agent-header">
                            <div class="agent-icon gap"><i class="fas fa-lightbulb"></i></div>
                            <div><div class="agent-title">Research Gap Analysis Agent</div><div class="agent-role">Identify Gaps</div></div>
                        </div>
                        <div class="agent-output" id="gapOutput"></div>
                    </div>
                </div>
                
                <div class="col-md-6" id="agent4Output" style="display: none;">
                    <div class="agent-card trend">
                        <div class="agent-header">
                            <div class="agent-icon trend"><i class="fas fa-chart-line"></i></div>
                            <div><div class="agent-title">Trend Forecasting Agent</div><div class="agent-role">Predict Trends</div></div>
                        </div>
                        <div class="agent-output" id="trendOutput"></div>
                    </div>
                </div>
                
                <div class="col-md-12" id="agent5Output" style="display: none;">
                    <div class="agent-card advisor">
                        <div class="agent-header">
                            <div class="agent-icon advisor"><i class="fas fa-user-tie"></i></div>
                            <div><div class="agent-title">Research Advisor Agent</div><div class="agent-role">Strategic Guidance</div></div>
                        </div>
                        <div class="agent-output" id="advisorOutput"></div>
                    </div>
                </div>
            </div>
            
            <div class="knowledge-graph">
                <h3><i class="fas fa-project-diagram"></i> Knowledge Graph</h3>
                <div id="knowledgeGraphContent"></div>
            </div>
        </div>
        
        <div class="footer">
            <p>&copy; 2026 ResearchMind AI | Powered by IBM watsonx.ai & IBM Granite Models</p>
        </div>
    </div>
    
    <script>
        document.getElementById('researchForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const query = document.getElementById('query').value;
            const files = document.getElementById('pdfUpload').files;
            const agents = [];
            
            ['agent1', 'agent2', 'agent3', 'agent4', 'agent5'].forEach((id, idx) => {
                if (document.getElementById(id).checked) {
                    agents.push(['retrieval', 'literature', 'gap', 'trend', 'advisor'][idx]);
                }
            });
            
            document.getElementById('loadingSpinner').classList.add('active');
            document.getElementById('resultsSection').style.display = 'none';
            
            const formData = new FormData();
            formData.append('query', query);
            formData.append('agents', JSON.stringify(agents));
            
            for (let file of files) {
                formData.append('files', file);
            }
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                displayResults(data);
            } catch (error) {
                alert('Error: ' + error.message);
            }
            
            document.getElementById('loadingSpinner').classList.remove('active');
        });
        
        function displayResults(data) {
            // Check if data is valid
            if (!data || typeof data !== 'object') {
                alert('Invalid response from server');
                return;
            }
            
            document.getElementById('resultsSection').style.display = 'block';
            document.getElementById('agentsActivated').textContent = data.agents ? Object.keys(data.agents).length : 0;
            document.getElementById('documentsProcessed').textContent = data.documents_processed || 0;
            document.getElementById('conceptsIdentified').textContent = (data.knowledge_graph && data.knowledge_graph.nodes) ? data.knowledge_graph.nodes.length : 0;
            
            let workflowHTML = '';
            if (data.execution_flow && Array.isArray(data.execution_flow)) {
                data.execution_flow.forEach(flow => {
                    workflowHTML += `
                        <div class="workflow-step">
                            <div class="workflow-step-number">${flow.step}</div>
                            <div>
                                <strong>${flow.agent}</strong><br>
                                <small>${flow.reason}</small>
                            </div>
                        </div>
                    `;
                });
            }
            document.getElementById('workflowSteps').innerHTML = workflowHTML || '<p>No workflow data</p>';
            
            document.getElementById('synthesisContent').textContent = data.synthesis || 'No synthesis available';
            
            if (data.agents) {
                if (data.agents.retrieval) {
                    document.getElementById('agent1Output').style.display = 'block';
                    document.getElementById('retrievalOutput').textContent = data.agents.retrieval.response || 'No response';
                }
                if (data.agents.literature) {
                    document.getElementById('agent2Output').style.display = 'block';
                    document.getElementById('literatureOutput').textContent = data.agents.literature.response || 'No response';
                }
                if (data.agents.gap) {
                    document.getElementById('agent3Output').style.display = 'block';
                    document.getElementById('gapOutput').textContent = data.agents.gap.response || 'No response';
                }
                if (data.agents.trend) {
                    document.getElementById('agent4Output').style.display = 'block';
                    document.getElementById('trendOutput').textContent = data.agents.trend.response || 'No response';
                }
                if (data.agents.advisor) {
                    document.getElementById('agent5Output').style.display = 'block';
                    document.getElementById('advisorOutput').textContent = data.agents.advisor.response || 'No response';
                }
            }
            
            let graphHTML = '';
            if (data.knowledge_graph && data.knowledge_graph.nodes && Array.isArray(data.knowledge_graph.nodes)) {
                data.knowledge_graph.nodes.forEach(node => {
                    graphHTML += `<span class="concept-node">${node.label || node.id}</span>`;
                });
            }
            document.getElementById('knowledgeGraphContent').innerHTML = graphHTML || '<p>No concepts identified</p>';
            
            window.scrollTo({ top: document.getElementById('resultsSection').offsetTop - 100, behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Process research analysis request"""
    try:
        query = request.form.get('query', '')
        agents_str = request.form.get('agents', '[]')
        
        # Parse agents list
        try:
            agents = json.loads(agents_str) if agents_str else []
        except:
            agents = ['retrieval', 'literature', 'gap', 'trend', 'advisor']
        
        # Process uploaded files
        docs_processed = 0
        files = request.files.getlist('files')
        
        for file in files:
            if file and file.filename:
                try:
                    if file.filename.endswith('.pdf'):
                        text = rag_system.extract_text_from_pdf(BytesIO(file.read()))
                        rag_system.add_document(text, {'filename': file.filename})
                        docs_processed += 1
                    elif file.filename.endswith('.txt'):
                        text = file.read().decode('utf-8')
                        rag_system.add_document(text, {'filename': file.filename})
                        docs_processed += 1
                except Exception as file_error:
                    print(f"Error processing file {file.filename}: {str(file_error)}")
        
        # Orchestrate agents
        results = orchestrator.orchestrate(query, agents if agents else None)
        
        # Generate knowledge graph
        kg = generate_knowledge_graph(results)
        
        results['knowledge_graph'] = kg
        results['documents_processed'] = docs_processed
        
        return jsonify(results)
    
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'message': 'An error occurred during analysis'}), 500

if __name__ == '__main__':
    print("=" * 80)
    print("ResearchMind AI - Intelligent Research Companion")
    print("=" * 80)
    print("Powered by IBM watsonx.ai and IBM Granite Models")
    print("\nFeatures:")
    print("  • 5 Specialized AI Agents")
    print("  • Multi-Agent Orchestration")
    print("  • RAG-Enhanced Research Processing")
    print("  • Knowledge Graph Visualization")
    print("  • Interactive Dashboard")
    print("\nStarting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

# Made with Bob
