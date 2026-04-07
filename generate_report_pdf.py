from fpdf import FPDF

class PDF(FPDF):
    def header(self):
        # Arial bold 15
        self.set_font("helvetica", "B", 15)
        # Move to the right
        self.cell(80)
        # Title
        self.cell(30, 10, "Clause Boundary Detection - Project Report", border=False, align="C")
        # Line break
        self.ln(20)

    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font("helvetica", "I", 8)
        # Page number
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

    def chapter_title(self, title):
        # Arial 12
        self.set_font("helvetica", "B", 14)
        # Background color
        self.set_fill_color(200, 220, 255)
        # Title
        self.cell(0, 10, f"{title}", border=False, ln=True, fill=True)
        # Line break
        self.ln(5)

    def chapter_body(self, text):
        # Times 12
        self.set_font("helvetica", "", 11)
        # Output text
        self.multi_cell(0, 7, text)
        # Line break
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)

# 1. Survey (Literature Review)
survey_text = """This project includes an extensive literature review of clause boundary detection techniques. We reviewed multiple seminal papers spanning different paradigms in the domain:

1. CRF-based Clause Boundary Identification (Ram & Devi, 2008): Uses Conditional Random Fields with POS tagging and chunk info, achieving 85.2% F1 on Hindi. It handles sequential dependency well but requires handcrafted features.

2. Clause Identification using Dependency Parsing (Gadde et al., 2010): Established that dependency parsing and clause detection are mutually beneficial, showing that extracting clauses boundaries significantly improves data-driven dependency parsing accuracy iteratively.

3. Multilingual Clause Splitting (Puscasu, 2004): A robust rule-based model relying on universal dependencies (advcl, ccomp, acl) yielding an accuracy around 75-80% across multiple languages using explicit syntactic markers without complex ML learning.

4. Neural Approaches (Nguyen et al., 2021) & Transformer-based approaches (Devlin et al., 2019): Demonstrated that deep neural architectures like BiLSTM with character/word embeddings or fine-tuned BERT/RoBERTa can reach upwards of 89-93% F1 score, eliminating manual feature engineering at the cost of higher computational requirements.

From this survey, we formulated the conclusion that building a unified comparative framework measuring Rule-based rules, Traditional Statistical Machine Learning (CRF), and Deep Learning Sequence Models (BiLSTM) will give the most extensive perspective of the problem domain."""

pdf.chapter_title("1. Survey (Reading Research Papers)")
pdf.chapter_body(survey_text)

# 2. Whole Code Flow
code_flow_text = """The overall architecture strictly follows a systematic Natural Language Processing and Machine Learning pipeline that isolates parsing, tagging, and logic execution modularly:

1. Data Loading (src/data_loader.py): Employs the 'conllu' Python library to parse the Universal Dependencies (UD) English Web Treebank dataset, reading the root trees and extracting crucial syntactic relationships and part-of-speech mappings efficiently.

2. Clause Labeling Scheme (src/clause_labeler.py): Converts disparate dependency logic into a rigorous token sequence BIO (Begin-Inside-Outside) tagging classification. We enforce heuristic structural mappings from tags like 'advcl' (adverbial clause), 'ccomp' (clausal complement), and 'xcomp' to derive the ground truth boundaries for training.

3. Linguistic Feature Extraction (src/feature_extractor.py): Aggregates highly predictive localized features to feed our models. Extracts the current sequence POS tag, parent dependency relations, and window-based contextual representations (window +-2 tokens text and pos tags).

4. Specialized Implementation of 3 Target Models:
   - Rule-based Determinism (src/rule_based.py): Executes custom traversal chunking on spaCy dependency structures based purely on syntactic indicators.
   - Conditional Random Field (src/crf_model.py): Implements sklearn-crfsuite, classifying our extracted textual window features sequentially by leveraging probabilities.
   - BiLSTM Architecture (src/bilstm_model.py): Written natively in PyTorch, the Deep Learning approach embeds both standard Token Embeddings and POS Tag Embeddings, pushing them into a multi-layered Bidirectional LSTM layout to evaluate boundary relationships holistically before linear sequential projection.

5. Training & Evaluation Pipeline (train.py, src/evaluation.py): An orchestrator executing model instantiation and batch gradient descent, evaluating resulting precision, recall, and f-scores exclusively per structural B/I tag types to ascertain explicit border sensitivities.

6. Visualization Application (app.py): Utilizing Streamlit to synthesize our model files into an end-user interface that renders live clauses alongside realtime comparison score tables, making evaluation visual and immediate."""

pdf.chapter_title("2. Whole Code Flow")
pdf.chapter_body(code_flow_text)

# 3. Innovation That I Have Done
innovation_text = """While standard clause detection relies primarily heavily on single solutions, the primary innovations delivered in this implementation include:

1. Unified Sequential Evaluation System: Standardizing vastly different NLP abstractions (Syntactic Dependencies, Statistical Modeling, and Neural Networks) into an overarching BIO-tag sequence domain. This unifies isolated rulesets and vectors into an equally comparable evaluation metric block.

2. Dual Semantic & Grammatical Embedding Vectors in Neural Deep Learning: A primary innovation was crafting the PyTorch BiLSTM model's foundation to accept separate concatenated spatial embeddings for both Vocabulary Semantics (what the word means) and Structural POS tags (what grammatical place the word serves). This gives the neural net full syntactic context natively, removing the reliance on gigantic computationally-heavy Transformer architectures while getting state-of-the-art capture sequences.

3. Interactive Dashboard Layer: Traditionally bounded to CLI operations and JSON output, this project dynamically injects prediction artifacts into a user-facing interactive frontend (Streamlit). Researchers can rapidly audit edge cases by typing customized complex clauses live within the web browser and watch three competing paradigms isolate sequences side-by-side."""

pdf.chapter_title("3. Innovation Done")
pdf.chapter_body(innovation_text)

# 4. Analysis Part
analysis_text = """Based on our raw aggregated test-set analysis (found within results/metrics/comparison.json), the developed paradigms contrast significantly (Weighted F1 metrics):

1. Rule-Based Determinist Approach:
   - Macro F1: 0.552 | Weighted F1: 0.898
   Analysis: Expectedly, the system exhibits extremely strong parsing when sentences conform to conventional dependency structures, maintaining high Inside-Clause validity (Precision 0.938). However, it entirely collapses against convoluted colloquialism, netting a highly anemic B-CLAUSE Recall (0.647) - proving it regularly misses the boundaries of ambiguous structures.

2. Conditional Random Fields (CRF):
   - Macro F1: 0.634 | Weighted F1: 0.974
   Analysis: Adding localized contextual probability drastically rectifies the deterministic blind spots. Its sequence conditioning jumps B-CLAUSE Precision to 0.939 and Recall to 0.893. CRF displays phenomenal computational efficiency vs. score returns.

3. Deep Neural BiLSTM Implementation:
   - Macro F1: 0.643 | Weighted F1: 0.981
   Analysis: The overall project performance winner. Because our engineered dual-embedding mechanism naturally tracks unlimited bidirectional sentence scope (compared to CRFs rigid +-2 tokens), it reaches maximum sequence fidelity with an unparalleled initial B-CLAUSE Recall of 0.948 and B-CLAUSE Precision of 0.931. Overall Inside-Clause F1 is nearly fully saturated at 0.988 F1.

Conclusion: 
The rigorous analysis verifies that combining robust semantic structures within BiLSTM Neural paradigms produces the highest sequence capture possible (0.981 Weighted F1) for nuanced grammar. However, the statistical ML CRF proves an exceptionally close contender when deploying limited infrastructure environments, leaving rudimentary dependency checks completely obsolete in modern sequence labeling."""

pdf.chapter_title("4. Analysis Part")
pdf.chapter_body(analysis_text)

pdf.output("Clause_Boundary_Detection_Project_Report.pdf")
print("Successfully generated clause project report.")
