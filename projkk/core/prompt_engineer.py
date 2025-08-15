"""
Prompt engineering module for Excel analysis.
Manages and optimizes prompts for different analysis tasks.
"""

import logging
from typing import Dict, List, Optional, Any
from config.settings import ANALYSIS_CATEGORIES, SYSTEM_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class PromptEngineer:
    """Manages and optimizes prompts for Excel analysis tasks."""
    
    def __init__(self):
        self.analysis_categories = ANALYSIS_CATEGORIES
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE
        self.prompt_templates = self._initialize_prompt_templates()
        
    def _initialize_prompt_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize prompt templates for different analysis types."""
        return {
            "summary": {
                "name": "Data Summary Analysis",
                "description": "Generate comprehensive data overview and key statistics",
                "template": """Analyze the Excel data and provide a comprehensive summary including:
1. Data overview (rows, columns, data types)
2. Key statistics for numerical columns
3. Missing data analysis
4. Data quality insights
5. Recommendations for data cleaning if needed

Focus on the most important patterns and insights that would be valuable for business decision-making."""
            },
            
            "trends": {
                "name": "Trend Analysis",
                "description": "Identify time-based patterns and trends in the data",
                "template": """Analyze the Excel data to identify trends and patterns:

1. Time-based trends (if date/time columns exist)
2. Sequential patterns in numerical data
3. Seasonal variations or cyclical patterns
4. Growth or decline trends
5. Anomalies or breakpoints in trends

Provide specific examples with data points and suggest appropriate visualizations (line charts, trend lines, etc.)."""
            },
            
            "anomalies": {
                "name": "Anomaly Detection",
                "description": "Identify outliers and unusual patterns in the data",
                "template": """Detect anomalies and outliers in the Excel data:

1. Statistical outliers using standard deviation or IQR methods
2. Unusual patterns or values that deviate from normal ranges
3. Data quality issues or inconsistencies
4. Potential errors or data entry problems
5. Recommendations for handling anomalies

Include specific examples with actual values and their statistical significance."""
            },
            
            "correlations": {
                "name": "Correlation Analysis",
                "description": "Analyze relationships between different variables",
                "template": """Analyze correlations and relationships between variables in the Excel data:

1. Strong positive and negative correlations
2. Weak or no correlations
3. Potential causal relationships
4. Confounding variables
5. Recommendations for further analysis

Provide correlation coefficients where applicable and suggest scatter plots or heatmaps for visualization."""
            },
            
            "insights": {
                "name": "Business Intelligence",
                "description": "Extract actionable business insights from the data",
                "template": """Extract actionable business insights from the Excel data:

1. Key performance indicators (KPIs)
2. Business opportunities or risks
3. Customer behavior patterns
4. Operational efficiency insights
5. Strategic recommendations

Focus on insights that can drive business decisions and include specific data points to support your analysis."""
            },
            
            "forecasting": {
                "name": "Predictive Analysis",
                "description": "Provide forecasting insights and future trends",
                "template": """Provide predictive analysis and forecasting insights:

1. Identify patterns that suggest future trends
2. Seasonal or cyclical forecasting
3. Growth projections based on historical data
4. Risk factors that could affect future outcomes
5. Confidence intervals and assumptions

Note: This is based on historical patterns and should be used as guidance, not definitive predictions."""
            },
            
            "custom": {
                "name": "Custom Analysis",
                "description": "Custom analysis based on user-specific questions",
                "template": """Analyze the Excel data to answer the specific user question:

{user_question}

Provide a comprehensive analysis that:
1. Directly addresses the question
2. Uses relevant data from the Excel file
3. Includes specific examples and data points
4. Suggests appropriate visualizations
5. Offers actionable insights or recommendations"""
            }
        }
    
    def get_analysis_prompt(self, analysis_type: str, user_question: str = None, **kwargs) -> Dict[str, str]:
        """Get a prompt for a specific analysis type."""
        try:
            if analysis_type not in self.prompt_templates:
                logger.warning(f"Unknown analysis type: {analysis_type}. Using custom analysis.")
                analysis_type = "custom"
            
            template = self.prompt_templates[analysis_type]
            
            if analysis_type == "custom" and user_question:
                prompt_text = template["template"].format(user_question=user_question)
            else:
                prompt_text = template["template"]
            
            # Add any additional context from kwargs
            if kwargs:
                context_parts = []
                for key, value in kwargs.items():
                    if value:
                        context_parts.append(f"{key}: {value}")
                
                if context_parts:
                    context_text = "\n\nAdditional Context:\n" + "\n".join(context_parts)
                    prompt_text += context_text
            
            return {
                "analysis_type": analysis_type,
                "name": template["name"],
                "description": template["description"],
                "prompt": prompt_text,
                "system_prompt": self.system_prompt
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis prompt: {str(e)}")
            return {
                "analysis_type": "error",
                "name": "Error",
                "description": "Failed to generate prompt",
                "prompt": f"Error: {str(e)}",
                "system_prompt": self.system_prompt
            }
    
    def create_enhanced_prompt(self, base_prompt: str, context: str, query: str) -> str:
        """Create an enhanced prompt by combining system prompt, context, and user query."""
        try:
            enhanced_prompt = f"""{self.system_prompt}

{base_prompt}

Context from Excel data:
{context}

User Question: {query}

Analysis:"""
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error creating enhanced prompt: {str(e)}")
            return base_prompt
    
    def optimize_prompt_for_model(self, prompt: str, model_name: str) -> str:
        """Optimize prompt for specific model characteristics."""
        try:
            optimized_prompt = prompt
            
            # Model-specific optimizations
            if "gpt" in model_name.lower():
                # GPT models work well with clear instructions
                optimized_prompt = f"Please provide a detailed analysis:\n\n{optimized_prompt}"
            
            elif "dialo" in model_name.lower():
                # DialoGPT models work better with conversational prompts
                optimized_prompt = f"Let me analyze this data for you:\n\n{optimized_prompt}"
            
            elif "bert" in model_name.lower() or "roberta" in model_name.lower():
                # BERT-based models prefer structured prompts
                optimized_prompt = f"Task: Data Analysis\n\n{optimized_prompt}"
            
            return optimized_prompt
            
        except Exception as e:
            logger.error(f"Error optimizing prompt: {str(e)}")
            return prompt
    
    def get_prompt_suggestions(self, data_summary: Dict) -> List[str]:
        """Generate prompt suggestions based on data characteristics."""
        try:
            suggestions = []
            
            # Based on data size
            total_rows = data_summary.get("total_rows", 0)
            total_cols = data_summary.get("total_columns", 0)
            
            if total_rows > 1000:
                suggestions.append("This is a large dataset. Consider focusing on key patterns and using sampling for detailed analysis.")
            
            if total_cols > 10:
                suggestions.append("Multiple columns detected. Consider correlation analysis to identify relationships between variables.")
            
            # Based on data types
            sheets = data_summary.get("sheets", {})
            for sheet_name, sheet_data in sheets.items():
                summary = sheet_data.get("summary", {})
                
                # Check for time series data
                dtypes = summary.get("dtypes", {})
                if any("datetime" in str(dtype) for dtype in dtypes.values()):
                    suggestions.append(f"Time series data detected in {sheet_name}. Consider trend analysis and seasonal patterns.")
                
                # Check for categorical data
                categorical_info = summary.get("categorical_info", {})
                if categorical_info:
                    suggestions.append(f"Categorical variables found in {sheet_name}. Consider frequency analysis and grouping insights.")
                
                # Check for numerical data
                numeric_stats = summary.get("numeric_stats", {})
                if numeric_stats:
                    suggestions.append(f"Numerical data available in {sheet_name}. Consider statistical analysis and outlier detection.")
            
            return suggestions[:5]  # Limit to 5 suggestions
            
        except Exception as e:
            logger.error(f"Error generating prompt suggestions: {str(e)}")
            return ["Consider analyzing the data structure and key patterns."]
    
    def validate_prompt(self, prompt: str) -> Dict[str, Any]:
        """Validate prompt quality and provide feedback."""
        try:
            validation_result = {
                "is_valid": True,
                "issues": [],
                "suggestions": [],
                "word_count": len(prompt.split()),
                "character_count": len(prompt)
            }
            
            # Check prompt length
            if len(prompt) < 50:
                validation_result["is_valid"] = False
                validation_result["issues"].append("Prompt is too short for meaningful analysis")
                validation_result["suggestions"].append("Add more specific instructions or context")
            
            if len(prompt) > 2000:
                validation_result["issues"].append("Prompt is very long and may confuse the model")
                validation_result["suggestions"].append("Consider breaking into smaller, focused prompts")
            
            # Check for specific instructions
            if "analyze" not in prompt.lower() and "examine" not in prompt.lower():
                validation_result["suggestions"].append("Consider adding specific action words like 'analyze', 'examine', or 'identify'")
            
            # Check for context
            if "data" not in prompt.lower() and "excel" not in prompt.lower():
                validation_result["suggestions"].append("Consider adding context about the data source")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating prompt: {str(e)}")
            return {
                "is_valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "suggestions": ["Check prompt format and try again"],
                "word_count": 0,
                "character_count": 0
            }
    
    def get_all_analysis_types(self) -> List[Dict[str, str]]:
        """Get all available analysis types with their descriptions."""
        return [
            {
                "type": analysis_type,
                "name": template["name"],
                "description": template["description"]
            }
            for analysis_type, template in self.prompt_templates.items()
        ]
    
    def create_comparison_prompt(self, analysis_types: List[str]) -> str:
        """Create a prompt for comparing multiple analysis types."""
        try:
            if not analysis_types:
                return "Please provide a general analysis of the Excel data."
            
            comparison_prompt = f"""Please provide a comprehensive analysis covering the following aspects:

{chr(10).join(f"{i+1}. {self.prompt_templates.get(analysis_type, {}).get('name', analysis_type)}" 
              for i, analysis_type in enumerate(analysis_types))}

For each aspect, provide:
- Key findings and insights
- Supporting data points
- Relevant visualizations
- Actionable recommendations

Ensure the analysis is well-structured and easy to follow."""
            
            return comparison_prompt
            
        except Exception as e:
            logger.error(f"Error creating comparison prompt: {str(e)}")
            return "Please provide a comprehensive analysis of the Excel data." 