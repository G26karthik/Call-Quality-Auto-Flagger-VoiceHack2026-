# Hackathon Dataset -- Data Dictionary

## Task
**Binary classification**: Predict `has_ticket` (True/False) for each call.
**Multi-label extension**: Predict which module categories have issues.

## Call Metadata
| Field | Type | Description |
|-------|------|-------------|
| call_id | UUID | Unique call identifier |
| outcome | string | Call outcome (completed, incomplete, opted_out, scheduled, escalated, wrong_number, etc.) |
| call_duration | int | Call duration in seconds |
| attempt_number | int | Which attempt this was (1 = first try) |
| direction | string | inbound or outbound |
| attempted_at | timestamp | When the call was placed (UTC) |
| scheduled_at | timestamp | When the call was scheduled for |
| billing_duration | int | Billing duration in seconds |
| pipeline_status | string | Pipeline verification status |
| pipeline_mismatch_count | int | Number of mismatches in pipeline verification |
| review_flag | bool | Whether the call was flagged for review |
| review_flag_reason | string | Reason for review flag |

## Context
| Field | Type | Description |
|-------|------|-------------|
| organization_id | UUID | Which organization this call belongs to |
| product_id | UUID | Which product/medication |
| patient_state | string | US state (for timezone context) |
| cycle_status | string | Case cycle status |
| form_submitted | bool | Whether the patient form was submitted |
| patient_name_anon | string | **Anonymized** patient name (fake) |

## Response Features
| Field | Type | Description |
|-------|------|-------------|
| question_count | int | Total questions in the call script |
| answered_count | int | How many questions got answers |
| response_completeness | float | answered_count / question_count |
| responses_json | JSON string | Full Q&A responses (anonymized) |
| validation_notes | string | Post-call validation notes |
| whisper_transcript | string | Whisper-verified transcript text |

## Transcript Features (from ElevenLabs)
| Field | Type | Description |
|-------|------|-------------|
| transcript_text | string | Full flattened transcript ([AGENT]: ... [USER]: ...) |
| turn_count | int | Total conversation turns |
| user_turn_count | int | Patient speaking turns |
| agent_turn_count | int | AI agent speaking turns |
| user_word_count | int | Total words spoken by patient |
| agent_word_count | int | Total words spoken by agent |
| avg_user_turn_words | float | Average words per patient turn |
| avg_agent_turn_words | float | Average words per agent turn |
| interruption_count | int | Times the conversation was interrupted |
| max_time_in_call | float | Last timestamp in transcript (seconds) |

## Time Features
| Field | Type | Description |
|-------|------|-------------|
| hour_of_day | int | Hour (0-23) when call was placed |
| day_of_week | string | Day name (Monday-Sunday) |

## Target Variables
| Field | Type | Description |
|-------|------|-------------|
| **has_ticket** | **bool** | **PRIMARY TARGET -- was a review ticket raised for this call?** |
| ticket_priority | string | urgent, high, normal, low |
| ticket_status | string | open, resolved, in_progress |
| ticket_initial_notes | string | Human reviewer notes on what was wrong |
| ticket_resolution_notes | string | How the ticket was resolved |

## Module-Level Categories
Each module is rated "success" (no issue) or "issue" (problem found).
| Field | Type | Description |
|-------|------|-------------|
| ticket_cat_audio_issue | string | Audio quality issues |
| ticket_cat_elevenlabs | string | ElevenLabs AI agent issues (misunderstood, wrong response) |
| ticket_cat_openai | string | OpenAI/LLM processing issues |
| ticket_cat_supabase | string | Database storage issues |
| ticket_cat_scheduler_aws | string | Scheduling/infrastructure issues |
| ticket_cat_other | string | Other issues |
| ticket_cat_*_notes | string | Detailed notes for each category |

## Common Ticket Reasons
- **Outcome miscategorization**: Call marked as wrong outcome (e.g., incomplete should be opted_out)
- **STT mishearing**: Speech-to-text misheard values (e.g., weight "62" instead of "262")
- **Agent behavior**: AI agent skipped questions, gave wrong info, or failed to escalate
- **Wrong number misclassification**: Patient said "not interested" but classified as wrong_number

## PII Note
All patient names have been replaced with fake names. Phone numbers are removed.
Real names in transcripts and notes have been redacted.
