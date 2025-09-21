# OpenRouter Integration Setup

This guide explains how to set up and use the OpenRouter API integration for AI-powered security briefings using Gemini 2.0 Flash.

## Setup Instructions

### 1. Get Your OpenRouter API Key

1. Visit [OpenRouter.ai](https://openrouter.ai/)
2. Sign up for an account
3. Go to [API Keys](https://openrouter.ai/keys)
4. Create a new API key
5. Copy the key for use in your environment

### 2. Configure Environment Variables

Create a `.env` file in your project root with the following content:

```bash
# OpenRouter API Configuration
OPENROUTER_API_KEY=your_actual_api_key_here

# Google API Key (for image descriptions - optional)
GOOGLE_API_KEY=your_google_api_key_here
```

**Important:** Replace `your_actual_api_key_here` with your actual OpenRouter API key.

### 3. Install Dependencies

Make sure you have all required dependencies installed:

```bash
# Activate your virtual environment first
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## How It Works

### AI Briefing Generation

When you click the "Generate Briefing" button on the dashboard:

1. **Data Collection**: The system reads all events from `events.json`
2. **AI Processing**: Events are sent to OpenRouter's Gemini 2.0 Flash model
3. **Analysis**: The AI analyzes patterns, critical incidents, and trends
4. **Report Generation**: A comprehensive security briefing is generated
5. **Display**: The formatted report is shown in the frontend

### Features

- **Professional Analysis**: AI acts as a security analyst
- **Pattern Recognition**: Identifies trends and busy periods
- **Critical Alert Highlighting**: Emphasizes high-priority events
- **Actionable Insights**: Provides recommendations
- **Formatted Output**: Clean, readable markdown formatting

### Model Used

- **Model**: `google/gemini-2.0-flash-001` via OpenRouter
- **Temperature**: 0.7 (balanced creativity and consistency)
- **Max Tokens**: 2000 (sufficient for comprehensive reports)

## API Endpoints

### POST /api/generate_briefing

Generates an AI-powered security briefing from all system events.

**Request**: No body required - automatically uses all events from `events.json`

**Response**:
```json
{
  "success": true,
  "briefing": "Formatted markdown briefing content...",
  "model_used": "google/gemini-2.0-flash-001"
}
```

## Error Handling

The system includes comprehensive error handling:

- **API Key Missing**: Clear error message if OpenRouter API key is not configured
- **No Events**: Graceful handling when no events exist
- **API Errors**: Detailed error messages from OpenRouter API
- **Network Issues**: Timeout and connection error handling

## Troubleshooting

### Common Issues

1. **"OpenRouter service not configured"**
   - Check that your `.env` file exists and contains `OPENROUTER_API_KEY`
   - Verify the API key is correct and active

2. **"No events found"**
   - Ensure your surveillance/monitoring systems are generating events
   - Check that `events.json` contains valid event data

3. **API Rate Limits**
   - OpenRouter has usage limits based on your plan
   - Consider upgrading if you hit rate limits

### Testing the Integration

You can test the OpenRouter connection by checking the console output when starting the application:

```
✅ OpenRouter service configured successfully.
```

If you see this message, the integration is working correctly.

## Cost Information

OpenRouter pricing for Gemini 2.0 Flash:
- Check current pricing at [OpenRouter Pricing](https://openrouter.ai/pricing)
- Briefing generation typically uses 1000-2000 tokens per request
- Monitor your usage in the OpenRouter dashboard

## Security Notes

- API keys are stored in environment variables, not in code
- The `.env` file should not be committed to version control
- OpenRouter requests include proper headers for analytics
- All AI processing happens through secure HTTPS connections
