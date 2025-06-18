#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-Model Email Agent
-----------------------
This script implements an automated sales email system using multiple LLM models,
agent orchestration, and guardrails.

It demonstrates:
1. Different models integration (OpenAI, DeepSeek, Google Gemini, Groq)
2. Structured outputs
3. Input guardrails
4. Web interface with Gradio
"""

from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, trace, function_tool, OpenAIChatCompletionsModel, input_guardrail, GuardrailFunctionOutput
from typing import Dict, List, Optional
import sendgrid
import os
import asyncio
import gradio as gr
from sendgrid.helpers.mail import Mail, Email, To, Content
from pydantic import BaseModel

# Load environment variables
load_dotenv(override=True)

# API Keys setup
openai_api_key = os.getenv('OPENAI_API_KEY')
google_api_key = os.getenv('GOOGLE_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')

# Global variables to store agent components
sales_manager = None
setup_complete = False
setup_status = "Not started"

def check_api_keys():
    """Check and print API key status"""
    status_messages = []
    status_messages.append("=== API Key Status ===")
    
    if openai_api_key:
        status_messages.append(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
    else:
        status_messages.append("OpenAI API Key not set")

    if google_api_key:
        status_messages.append(f"Google API Key exists and begins {google_api_key[:2]}")
    else:
        status_messages.append("Google API Key not set (and this is optional)")

    if deepseek_api_key:
        status_messages.append(f"DeepSeek API Key exists and begins {deepseek_api_key[:3]}")
    else:
        status_messages.append("DeepSeek API Key not set (and this is optional)")

    if groq_api_key:
        status_messages.append(f"Groq API Key exists and begins {groq_api_key[:4]}")
    else:
        status_messages.append("Groq API Key not set (and this is optional)")
    
    status_messages.append("=====================")
    
    # Print to console and return as string for UI
    print("\n".join(status_messages))
    return "\n".join(status_messages)

# Define sales agent instructions
instructions1 = """You are a sales agent working for ComplAI, 
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
You write professional, serious cold emails."""

instructions2 = """You are a humorous, engaging sales agent working for ComplAI, 
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
You write witty, engaging cold emails that are likely to get a response."""

instructions3 = """You are a busy sales agent working for ComplAI, 
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI. 
You write concise, to the point cold emails."""

# Set up model endpoints
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Create API clients
def setup_clients():
    """Set up API clients for different models"""
    print("Setting up API clients for multiple LLM providers...")
    
    # Check if OpenAI API key is available (required)
    if not openai_api_key:
        raise ValueError("OpenAI API key is required for the application to function")
    
    # Initialize models list
    models = []
    
    # Setup DeepSeek if available
    if deepseek_api_key:
        deepseek_client = AsyncOpenAI(base_url=DEEPSEEK_BASE_URL, api_key=deepseek_api_key)
        deepseek_model = OpenAIChatCompletionsModel(model="deepseek-chat", openai_client=deepseek_client)
        models.append(("deepseek", deepseek_model))
    else:
        print("DeepSeek API key not available, using OpenAI fallback")
        # Create fallback using OpenAI
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        fallback_model1 = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client)
        models.append(("deepseek_fallback", fallback_model1))
    
    # Setup Gemini if available
    if google_api_key:
        gemini_client = AsyncOpenAI(base_url=GEMINI_BASE_URL, api_key=google_api_key)
        gemini_model = OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=gemini_client)
        models.append(("gemini", gemini_model))
    else:
        print("Google API key not available, using OpenAI fallback")
        # Create fallback using OpenAI
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        fallback_model2 = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client)
        models.append(("gemini_fallback", fallback_model2))
    
    # Setup Groq if available
    if groq_api_key:
        groq_client = AsyncOpenAI(base_url=GROQ_BASE_URL, api_key=groq_api_key)
        llama3_3_model = OpenAIChatCompletionsModel(model="llama-3.3-70b-versatile", openai_client=groq_client)
        models.append(("llama", llama3_3_model))
    else:
        print("Groq API key not available, using OpenAI fallback")
        # Create fallback using OpenAI
        openai_client = AsyncOpenAI(api_key=openai_api_key)
        fallback_model3 = OpenAIChatCompletionsModel(model="gpt-4o-mini", openai_client=openai_client)
        models.append(("llama_fallback", fallback_model3))
    
    # Return the models
    return models[0][1], models[1][1], models[2][1]

# Create sales agents
def create_sales_agents(deepseek_model, gemini_model, llama3_3_model):
    """Create sales agents with different models and personalities"""
    print("Creating sales agents with different personalities...")
    sales_agent1 = Agent(name="DeepSeek Sales Agent", instructions=instructions1, model=deepseek_model)
    sales_agent2 = Agent(name="Gemini Sales Agent", instructions=instructions2, model=gemini_model)
    sales_agent3 = Agent(name="Llama3.3 Sales Agent", instructions=instructions3, model=llama3_3_model)
    
    description = "Write a cold sales email"
    
    tool1 = sales_agent1.as_tool(tool_name="sales_agent1", tool_description=description)
    tool2 = sales_agent2.as_tool(tool_name="sales_agent2", tool_description=description)
    tool3 = sales_agent3.as_tool(tool_name="sales_agent3", tool_description=description)
    
    return [tool1, tool2, tool3]

# Email sending function tool
@function_tool
def send_html_email(subject: str, html_body: str) -> Dict[str, str]:
    """Send out an email with the given subject and HTML body to all sales prospects"""
    try:
        sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
        from_email = Email("your_verified_sender@example.com")  # Change to your verified sender
        to_email = To("your_recipient@example.com")  # Change to your recipient
        content = Content("text/html", html_body)
        mail = Mail(from_email, to_email, subject, content).get()
        response = sg.client.mail.send.post(request_body=mail)
        return {"status": "success"}
    except Exception as e:
        print(f"Email sending disabled or failed: {str(e)}")
        return {"status": "preview_only", "subject": subject, "html_body": html_body}

# Create email formatting agents
def create_email_tools():
    """Create agents for email subject and HTML conversion"""
    print("Creating email formatting tools...")
    subject_instructions = """You can write a subject for a cold sales email. 
    You are given a message and you need to write a subject for an email that is likely to get a response."""

    html_instructions = """You can convert a text email body to an HTML email body. 
    You are given a text email body which might have some markdown 
    and you need to convert it to an HTML email body with simple, clear, compelling layout and design."""

    subject_writer = Agent(name="Email subject writer", instructions=subject_instructions, model="gpt-4o-mini")
    subject_tool = subject_writer.as_tool(tool_name="subject_writer", tool_description="Write a subject for a cold sales email")

    html_converter = Agent(name="HTML email body converter", instructions=html_instructions, model="gpt-4o-mini")
    html_tool = html_converter.as_tool(tool_name="html_converter", tool_description="Convert a text email body to an HTML email body")
    
    return [subject_tool, html_tool, send_html_email]

# Create email manager agent
def create_email_manager(email_tools):
    """Create an agent to format and send emails"""
    print("Creating email manager agent...")
    instructions = """You are an email formatter and sender. You receive the body of an email to be sent. 
    You first use the subject_writer tool to write a subject for the email, then use the html_converter tool to convert the body to HTML. 
    Finally, you use the send_html_email tool to send the email with the subject and HTML body."""

    emailer_agent = Agent(
        name="Email Manager",
        instructions=instructions,
        tools=email_tools,
        model="gpt-4o-mini",
        handoff_description="Convert an email to HTML and send it"
    )
    
    return emailer_agent

# Create name check guardrail
def create_name_guardrail():
    """Create a guardrail to check for personal names"""
    print("Creating name checking guardrail...")
    class NameCheckOutput(BaseModel):
        is_name_in_message: bool
        name: str

    guardrail_agent = Agent(
        name="Name check",
        instructions="Check if the user is including someone's personal name in what they want you to do.",
        output_type=NameCheckOutput,
        model="gpt-4o-mini"
    )
    
    @input_guardrail
    async def guardrail_against_name(ctx, agent, message):
        result = await Runner.run(guardrail_agent, message, context=ctx.context)
        is_name_in_message = result.final_output.is_name_in_message
        return GuardrailFunctionOutput(
            output_info={"found_name": result.final_output},
            tripwire_triggered=is_name_in_message
        )
    
    return guardrail_against_name

# Create protected sales manager
def create_sales_manager(tools, handoffs, guardrail):
    """Create a sales manager with input guardrails"""
    print("Creating sales manager with guardrails...")
    sales_manager_instructions = """You are a sales manager working for ComplAI. You use the tools given to you to generate cold sales emails. 
    You never generate sales emails yourself; you always use the tools. 
    You try all 3 sales agent tools at least once before choosing the best one. 
    You can use the tools multiple times if you're not satisfied with the results from the first try. 
    You select the single best email using your own judgement of which email will be most effective. 
    After picking the email, you handoff to the Email Manager agent to format and send the email."""

    sales_manager = Agent(
        name="Sales Manager",
        instructions=sales_manager_instructions,
        tools=tools,
        handoffs=handoffs,
        model="gpt-4o-mini",
        input_guardrails=[guardrail]
    )
    
    return sales_manager

# Gradio interface functions
async def initialize_agents():
    """Initialize all agent components"""
    global sales_manager, setup_complete, setup_status
    
    try:
        setup_status = "Checking API keys..."
        check_api_keys()
        
        setup_status = "Setting up model clients..."
        deepseek_model, gemini_model, llama3_3_model = setup_clients()
        
        setup_status = "Creating sales agents..."
        sales_tools = create_sales_agents(deepseek_model, gemini_model, llama3_3_model)
        
        setup_status = "Creating email tools..."
        email_tools = create_email_tools()
        
        setup_status = "Creating email manager..."
        emailer_agent = create_email_manager(email_tools)
        
        setup_status = "Creating guardrails..."
        name_guardrail = create_name_guardrail()
        
        setup_status = "Creating sales manager..."
        sales_manager = create_sales_manager(sales_tools, [emailer_agent], name_guardrail)
        
        setup_complete = True
        setup_status = "Setup complete! Ready to generate emails."
        return "All agents initialized successfully!"
    except Exception as e:
        setup_status = f"Error during setup: {str(e)}"
        return f"Error initializing agents: {str(e)}"

async def generate_email(message, recipient_title, sender_title):
    """Generate an email based on user input"""
    global sales_manager, setup_complete
    
    if not setup_complete:
        return "System not initialized. Please click 'Initialize System' first."
    
    # Format the message with recipient and sender
    full_message = f"Send out a cold sales email addressed to Dear {recipient_title} from {sender_title}"
    if message:
        full_message += f" including these points: {message}"
    
    # Run the email generation with trace
    async with trace("Email Generation UI") as ctx:
        async for chunk in Runner.stream(sales_manager, full_message, context=ctx):
            yield chunk

def check_setup_status():
    """Check the current setup status"""
    global setup_status
    return setup_status

# Command line demo
async def run_demo():
    """Run a demonstration of the multi-model email agent"""
    # Check API keys
    check_api_keys()
    
    # Setup clients and models
    deepseek_model, gemini_model, llama3_3_model = setup_clients()
    
    # Create sales agents
    sales_tools = create_sales_agents(deepseek_model, gemini_model, llama3_3_model)
    
    # Create email tools
    email_tools = create_email_tools()
    
    # Create email manager
    emailer_agent = create_email_manager(email_tools)
    
    # Create name guardrail
    name_guardrail = create_name_guardrail()
    
    # Create protected sales manager
    sales_manager = create_sales_manager(sales_tools, [emailer_agent], name_guardrail)
    
    print("This demonstration will show how the system handles a request with a title instead of a personal name")
    message = "Send out a cold sales email addressed to Dear CEO from Head of Business Development"
    
    print("\nRunning sales manager with guardrails...")
    with trace("Complete Multi-Model Email Agent Demo"):
        result = await Runner.run(sales_manager, message)
    
    print("Check the execution trace at: https://platform.openai.com/traces")
    print("\nThe system successfully:")
    print("1. Checked for personal names in the request (guardrail)")
    print("2. Generated emails using three different LLM models")
    print("3. Selected the best email based on effectiveness")
    print("4. Formatted the email with a compelling subject and HTML body")

# Create the Gradio interface
def create_gradio_interface():
    """Create and return the Gradio interface"""
    with gr.Blocks(title="Multi-Model Email Agent") as demo:
        gr.Markdown("# Multi-Model Email Agent")
        gr.Markdown("""
        This application generates sales emails using multiple LLM models and selects the best one.
        It uses DeepSeek, Google Gemini, and Llama 3.3 models with different writing styles.
        
        ### How it works:
        1. First, click 'Initialize System' to set up all the agents
        2. Then, fill in the form and click 'Generate Email'
        3. The system will generate emails using three different models and select the best one
        """)
        
        with gr.Row():
            with gr.Column():
                init_button = gr.Button("Initialize System", variant="primary")
                status = gr.Textbox(label="System Status", value="Not initialized", interactive=False)
                # Fix: Use the async function directly without asyncio.create_task
                init_button.click(
                    fn=initialize_agents,
                    outputs=status
                )
                
                # Add a refresh button for status
                refresh_button = gr.Button("Refresh Status")
                refresh_button.click(
                    fn=check_setup_status,
                    outputs=status
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Email Configuration")
                recipient_title = gr.Textbox(
                    label="Recipient Title", 
                    placeholder="CEO, CTO, Head of Compliance",
                    value="CEO"
                )
                sender_title = gr.Textbox(
                    label="Sender Title", 
                    placeholder="Head of Sales, Business Development Manager",
                    value="Head of Business Development"
                )
                message = gr.Textbox(
                    label="Additional Context (optional)", 
                    placeholder="Any specific points you want to include in the email",
                    lines=3
                )
                generate_button = gr.Button("Generate Email", variant="primary")
            
        output = gr.Markdown(label="Generated Email")
        
        generate_button.click(
            fn=generate_email,
            inputs=[message, recipient_title, sender_title],
            outputs=output
        )
    
    return demo

if __name__ == "__main__":
    print("=== Multi-Model Email Agent ===")
    print("1. Command-line demo")
    print("2. Web interface")
    choice = input("Choose an option (1/2): ")
    
    if choice == "1":
        # Run the command-line demo
        asyncio.run(run_demo())
    else:
        # Create and launch the Gradio interface
        demo = create_gradio_interface()
        demo.launch() 