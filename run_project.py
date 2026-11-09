"""Run completed capstone sections in order."""
from src.pandas_analysis import run_pandas_analysis
from src.sql_analysis import run_sql_analysis
from src.data_quality_audit import run_audit
from src.clean_customer_orders import run_data_cleaning_pipeline
from src.validate_data import validate_cleaned_data

from src.customer_segmentation import run_customer_segmentation
from src.lapse_prediction import run_lapse_dataset
from src.lapse_preprocessing import run_lapse_preprocessing
from src.lapse_model_training import run_lapse_model_training
from src.lapse_class_balancing import run_class_balancing
from src.lapse_model_interpretation import run_model_interpretation

# Part C — Support ticket classification
from src.ticket_guardrails import run_ticket_guardrails
from src.ticket_redaction import run_ticket_redaction
from src.ticket_prompts import run_ticket_prompts
from src.ticket_prompt_comparison import run_ticket_prompt_comparison
from src.ticket_triage import run_ticket_triage

"""Run Part D: Policy RAG Assistant and Refund Agent."""
from src.policy_evaluation import run_policy_evaluation
from src.refund_agent import run_all_scenarios
from src.refund_safety_demo import main as run_safety_demo
from src.refund_resume_demo import main as run_resume_demo

def main():
     """Execute the capstone pipeline in order."""

     # Part A.Q2: Audit raw datasets
     run_audit()

     # Part A.Q3–A.Q5: Clean and store datasets
     run_data_cleaning_pipeline()

     # Part A.Q5: Validate saved outputs
     validate_cleaned_data() 

     # A.Q6 - SQL analysis
     run_sql_analysis()

     # A.Q7 - Pandas analysis and EDA
     run_pandas_analysis ()


     # ==========================================
     # PART B — CUSTOMER SEGMENTS AND LAPSE
     # ==========================================

     # Part B.Q1–B.Q3: RFM, K-means clustering,
     # customer segment profiles and PCA
     run_customer_segmentation()

     # Part B.Q4: Lapse dataset and baseline
     run_lapse_dataset()

     # Part B.Q5: Leakage-free preprocessing
     run_lapse_preprocessing()

     # Part B.Q6: Train, tune and evaluate models
     run_lapse_model_training()

     # Part B.Q7: Training-only class balancing
     run_class_balancing()

     # Part B.Q8: Feature interpretation, model
     # persistence and Premium retention list
     run_model_interpretation()

     # =========================================================
     # PART C — SUPPORT TICKET TRIAGE
     # =========================================================
     print("\n========== PART C ==========")

     # C.Q1 — Input guardrails
     tickets = run_ticket_guardrails()

     # C.Q2 — PII redaction
     redacted_tickets = run_ticket_redaction(tickets)

     # C.Q3 — Zero-shot and few-shot prompt definitions
     run_ticket_prompts()

     # C.Q4, C.Q5, C.Q6 — Retry, parsing and prompt comparison
     # These stages run together inside the comparison function.
     zero_results, few_results = run_ticket_prompt_comparison(
          redacted_tickets
     )

     # C.Q7(a) — Zero-shot budget demonstration
     # C.Q7(b) — Final few-shot triage and routing
     # Both CSV files are generated and validated internally.
     budget_demo, final_triage = run_ticket_triage(
          redacted_tickets
     )

     print("\nPART C COMPLETED")

     print("\n" + "=" * 60)
     print("PART D — POLICY RAG ASSISTANT AND REFUND AGENT")
     print("=" * 60)

     # D.Q1–D.Q4: Policy retrieval, grounded answers and evaluation.
     print("\nD.Q1–D.Q4: Policy RAG Evaluation")
     run_policy_evaluation()

     # D.Q5–D.Q6: Refund tools and seven ReAct scenarios.
     # D.Q5 tools are exercised through the agent.
     print("\nD.Q5–D.Q6: ReAct Refund Agent")
     run_all_scenarios()

     # D.Q7: Maximum iterations, token limit and timeout.
     print("\nD.Q7: Agent Safety Demonstrations")
     run_safety_demo()

     # D.Q8: Simulated crash and checkpoint recovery.
     print("\nD.Q8: Checkpoint and Resume")
     run_resume_demo()

     # D.Q9 is a written comparison documented in README.md.
     print("\nD.Q9: ReAct versus Plan-and-Execute")
     print("See README.md, Part D.Q9.")

     print("\nPART D EXECUTION COMPLETED")

if __name__ == "__main__":
    main()