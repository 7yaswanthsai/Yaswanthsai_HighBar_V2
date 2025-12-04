# Creative Generator Prompt

## Task
Produce new creative messages for campaigns identified as low-CTR.
Ground the recommendations in the dataset’s existing `creative_message` and `creative_type`.

## Reasoning Flow
Think → Analyze → Conclude  
- Think: Identify core themes and useful phrases.  
- Analyze: Blend phrases into new message structures with stronger CTAs.  
- Conclude: Provide diverse, fresh suggestions.

## Output Format
```json
{
  "campaign_name": {
    "suggestions": [
      "string"
    ],
    "source_examples": [
      "string"
    ]
  }
}
```

## Guidelines
- Use dataset phrasing where possible  
- Include CTA variations ("Shop Now", "Get Yours")  
- Emphasize benefits, value, or comfort cues  

## Reflection
If suggestions seem repetitive:
- Add more phrase diversity  
- Recombine extracted keyword sets  
