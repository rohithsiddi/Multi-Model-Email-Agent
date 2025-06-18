# Multi-Model Email Agent

A sophisticated automated sales email system that leverages multiple LLM models, agent orchestration, and guardrails.

## Features

- **Multiple Model Integration**: Seamlessly integrates with OpenAI, DeepSeek, Google Gemini, and Groq (Llama 3.3) models
- **Agent Orchestration**: Coordinates multiple specialized agents to generate, format, and send emails
- **Guardrails Implementation**: Includes input guardrails to prevent using personal names
- **Structured Output**: Uses Pydantic models for structured data validation
- **Web Interface**: User-friendly Gradio web UI for easy interaction

## Architecture

The system consists of several coordinated agents:

1. **Sales Agents**: Three different agents with distinct personalities:
   - Professional/serious (DeepSeek)
   - Humorous/engaging (Gemini)
   - Concise/to-the-point (Llama 3.3)

2. **Email Processing Agents**:
   - Subject Writer: Creates compelling email subjects
   - HTML Converter: Transforms plain text into formatted HTML
   - Email Sender: Handles email delivery via SendGrid

3. **Orchestration Agents**:
   - Sales Manager: Coordinates the sales agents and selects the best email
   - Email Manager: Handles formatting and sending

4. **Guardrail Agent**:
   - Name Check: Prevents using personal names in emails

## How It Works

When you run the application:

1. **Setup Phase**:
   - Loads API keys from environment variables
   - Initializes connections to multiple LLM providers
   - Creates specialized agents for different tasks

2. **User Input Phase**:
   - You specify the recipient title (e.g., "CEO")
   - You specify the sender title (e.g., "Head of Business Development")
   - You can add optional context for the email

3. **Execution Phase**:
   - Checks for personal names using the guardrail
   - Generates email variants using three different LLM models
   - Selects the best email based on effectiveness
   - Creates a compelling subject line
   - Formats the email as HTML
   - Displays the result in the UI (or sends it if SendGrid is configured)

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_key (required)
   GOOGLE_API_KEY=your_gemini_key (optional)
   DEEPSEEK_API_KEY=your_deepseek_key (optional)
   GROQ_API_KEY=your_groq_key (optional)
   SENDGRID_API_KEY=your_sendgrid_key (optional, required for sending emails)
   ```
   Note: 
   - OpenAI API key is required for the application to function
   - If optional API keys are missing, the system will fall back to using OpenAI models
   - SendGrid API key is required only if you want to actually send emails (otherwise, emails will be displayed in preview mode)

4. Update the email addresses in the `send_html_email` function in `multi_model_email_agent.py`:
   ```python
   from_email = Email("your_verified_sender@example.com")  # Change to your verified sender
   to_email = To("your_recipient@example.com")  # Change to your recipient
   ```

## Usage

### Web Interface

Run the application and select option 2 for the web interface:
```
python multi_model_email_agent.py
```

This will launch a Gradio web interface where you can:
1. Click "Initialize System" to set up all agents
2. Enter recipient and sender titles
3. Add optional context for the email
4. Click "Generate Email" to create your email

### Command Line

Alternatively, run the script and select option 1:
```
python multi_model_email_agent.py
```

This will run a demonstration with predefined inputs.

## Extending the System

You can extend this system by:
- Adding more LLM providers
- Creating additional guardrails
- Implementing more specialized agents
- Enhancing the email formatting capabilities
- Adding more features to the web interface

## Requirements

- Python 3.8+
- OpenAI API key (required)
- Google Gemini, DeepSeek, Groq API keys (optional)
- SendGrid API key (optional, for sending emails)
- Internet connection for API access 