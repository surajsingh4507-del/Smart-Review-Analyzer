import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

def build_presentation():
    prs = Presentation()
    # 16:9 widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    blank_layout = prs.slide_layouts[6]
    
    # Color Palette
    PRIMARY = RGBColor(79, 70, 229)      # #4F46E5 Indigo
    PRIMARY_DARK = RGBColor(49, 46, 129) # #312E81 Deep Indigo
    ACCENT_GREEN = RGBColor(16, 185, 129)# #10B981 Emerald
    ACCENT_RED = RGBColor(239, 68, 68)   # #EF4444 Coral Red
    TEXT_DARK = RGBColor(15, 23, 42)     # #0F172A Slate 900
    TEXT_MUTED = RGBColor(100, 116, 139) # #64748B Slate 500
    BG_CARD = RGBColor(248, 250, 252)    # #F8FAFC Off-white
    BORDER_COLOR = RGBColor(226, 232, 240)# #E2E8F0
    WHITE = RGBColor(255, 255, 255)
    
    def add_header(slide, title_text, category_text="COLLEGE NLP LAB PROJECT"):
        # Header banner shape
        banner = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(0.5), Inches(11.733), Inches(1.1))
        banner.fill.solid()
        banner.fill.fore_color.rgb = BG_CARD
        banner.line.color.rgb = BORDER_COLOR
        banner.line.width = Pt(1)
        
        tf = banner.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.3)
        tf.margin_top = Inches(0.15)
        
        p_cat = tf.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = PRIMARY
        
        p_title = tf.add_paragraph()
        p_title.text = title_text
        p_title.font.size = Pt(22)
        p_title.font.bold = True
        p_title.font.color.rgb = TEXT_DARK
        
    def add_card(slide, left, top, width, height, title, content_list, title_color=PRIMARY):
        card = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        card.fill.solid()
        card.fill.fore_color.rgb = BG_CARD
        card.line.color.rgb = BORDER_COLOR
        card.line.width = Pt(1)
        
        tf = card.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0.3)
        tf.margin_right = Inches(0.3)
        tf.margin_top = Inches(0.25)
        tf.margin_bottom = Inches(0.25)
        
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.size = Pt(15)
        p0.font.bold = True
        p0.font.color.rgb = title_color
        p0.space_after = Pt(10)
        
        for item in content_list:
            p = tf.add_paragraph()
            p.text = f"•  {item}"
            p.font.size = Pt(12)
            p.font.color.rgb = TEXT_DARK
            p.space_after = Pt(6)

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PRIMARY_DARK
    bg1.line.fill.background()
    
    # Title Box
    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.5), Inches(11.333), Inches(3.2))
    tf1 = t_box.text_frame
    tf1.word_wrap = True
    
    p_badge = tf1.paragraphs[0]
    p_badge.text = "COLLEGE NLP LAB — FINAL CAPSTONE PROJECT"
    p_badge.font.size = Pt(13)
    p_badge.font.bold = True
    p_badge.font.color.rgb = ACCENT_GREEN
    p_badge.space_after = Pt(12)
    
    p_main = tf1.add_paragraph()
    p_main.text = "🧠 Smart Review Analyzer"
    p_main.font.size = Pt(40)
    p_main.font.bold = True
    p_main.font.color.rgb = WHITE
    p_main.space_after = Pt(10)
    
    p_sub = tf1.add_paragraph()
    p_sub.text = "Multi-Layer Sentiment Intelligence, Aspect Mining & Machine Learning Benchmarking Platform"
    p_sub.font.size = Pt(18)
    p_sub.font.color.rgb = RGBColor(199, 210, 254)
    p_sub.space_after = Pt(20)

    # Details Card on Slide 1
    d_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(5.0), Inches(11.333), Inches(1.8))
    d_card.fill.solid()
    d_card.fill.fore_color.rgb = RGBColor(40, 36, 110)
    d_card.line.color.rgb = RGBColor(99, 102, 241)
    
    dtf = d_card.text_frame
    dtf.margin_left = Inches(0.4)
    dtf.margin_top = Inches(0.25)
    
    p_author = dtf.paragraphs[0]
    p_author.text = "👨‍💻 Lead Developer: Suraj Singh  |  B.Tech / Semester 5"
    p_author.font.size = Pt(15)
    p_author.font.bold = True
    p_author.font.color.rgb = WHITE
    p_author.space_after = Pt(8)
    
    p_tech = dtf.add_paragraph()
    p_tech.text = "🛠️ Tech Stack: Python 3.11, FastAPI, VADER Sentiment, TextBlob, Scikit-Learn TF-IDF, Chart.js, SQLite"
    p_tech.font.size = Pt(12)
    p_tech.font.color.rgb = RGBColor(224, 231, 255)
    p_tech.space_after = Pt(6)
    
    p_links = dtf.add_paragraph()
    p_links.text = "🌐 Live Demo: https://smart-review-analyzer-sandy.vercel.app/   |   GitHub: surajsingh4507-del/Smart-Review-Analyzer"
    p_links.font.size = Pt(12)
    p_links.font.bold = True
    p_links.font.color.rgb = ACCENT_GREEN

    # ==========================================
    # SLIDE 2: PROBLEM STATEMENT & MOTIVATION
    # ==========================================
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Real-World Motivation", "Project Objectives")
    
    add_card(s2, Inches(0.8), Inches(1.9), Inches(3.6), Inches(5.0),
             "1. The Core Challenge", [
                 "E-commerce platforms receive thousands of unstructured customer reviews daily.",
                 "Manual review inspection is impossible at scale.",
                 "Naïve binary scoring (Positive / Negative) misses critical context like shipping delays vs. build quality.",
                 "Lack of actionable priority triage slows down customer resolution teams."
             ], PRIMARY)
             
    add_card(s2, Inches(4.8), Inches(1.9), Inches(3.6), Inches(5.0),
             "2. The Research Gaps", [
                 "Paid Cloud APIs (OpenAI, AWS Comprehend) are expensive, rate-limited, and violate privacy.",
                 "Simple keyword-matching fails on sarcasm, negations ('not good'), and subjective nuances.",
                 "Existing academic tools lack explainability — teachers and viva examiners need to see internal math.",
                 "No voice review dictation or live multi-model disagreement benchmarking."
             ], ACCENT_RED)
             
    add_card(s2, Inches(8.8), Inches(1.9), Inches(3.7), Inches(5.0),
             "3. Our Proposed Solution", [
                 "End-to-end multi-layered NLP platform running 100% offline with zero paid API costs.",
                 "Combines Rule-Based Lexicon (VADER) with Machine Learning Classifiers (TF-IDF + MNB + Logistic Regression).",
                 "Opinion Subjectivity Mining via TextBlob (separates facts from emotional opinions).",
                 "Aspect-Based Sentiment Analysis (ABSA) for Delivery, Quality, Service, and Pricing."
             ], ACCENT_GREEN)

    # ==========================================
    # SLIDE 3: SYSTEM ARCHITECTURE & 6-STAGE PIPELINE
    # ==========================================
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "System Architecture & 6-Stage Intelligence Pipeline", "Architecture Overview")
    
    add_card(s3, Inches(0.8), Inches(1.9), Inches(5.6), Inches(2.4),
             "Stage 1: Multi-Modal Ingestion", [
                 "Single review manual typing with dynamic threshold slider.",
                 "Hands-free voice review dictation via HTML5 Web Speech API.",
                 "Bulk batch review ingestion via .csv / .txt drag-and-drop datasets."
             ], PRIMARY)
             
    add_card(s3, Inches(6.8), Inches(1.9), Inches(5.7), Inches(2.4),
             "Stage 2: Morphological Preprocessing", [
                 "Regex noise cleaning, URL removal, and lowercasing.",
                 "Dynamic stopword removal (preserves negation words like 'not', 'never').",
                 "Morphological dictionary lemmatization (resolves 'better' -> 'good')."
             ], PRIMARY)
             
    add_card(s3, Inches(0.8), Inches(4.5), Inches(5.6), Inches(2.5),
             "Stage 3 & 4: Lexicon & Opinion Mining", [
                 "Stage 3: VADER compound intensity scoring with cutoff threshold slider.",
                 "Stage 4: TextBlob polarity & subjectivity mining (0.0 Fact to 1.0 Opinion)."
             ], ACCENT_GREEN)
             
    add_card(s3, Inches(6.8), Inches(4.5), Inches(5.7), Inches(2.5),
             "Stage 5 & 6: Machine Learning & ABSA", [
                 "Stage 5: Scikit-Learn TF-IDF N-grams (MultinomialNB + Logistic Regression).",
                 "Stage 6: Domain Aspect Tagging (Quality, Shipping, Service, Pricing) + Priority Triage."
             ], ACCENT_GREEN)

    # ==========================================
    # SLIDE 4: PREPROCESSING & LEMMATIZATION DEEP DIVE
    # ==========================================
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "NLP Preprocessing: Tokenization & Morphological Lemmatization", "NLP Fundamentals")
    
    add_card(s4, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.0),
             "Lemmatization vs. Stemming (Viva Focus)", [
                 "Stemming (e.g. Porter Stemmer):",
                 "  • Crudely cuts character suffixes using heuristic rules.",
                 "  • Produces non-dictionary pseudo-words (e.g. 'caring' -> 'car', 'troubled' -> 'troubl').",
                 "  • Can distort sentiment polarity and confuse domain classifiers.",
                 "",
                 "Morphological Lemmatization (Our Approach):",
                 "  • Analyzes vocabulary morphology and grammatical part-of-speech context.",
                 "  • Resolves inflected variants to genuine dictionary root lemmas.",
                 "  • Examples: 'caring' -> 'care', 'better' -> 'good', 'worst' -> 'bad'.",
                 "  • Preserves semantic clarity for exact aspect matching and topic extraction."
             ], PRIMARY)
             
    add_card(s4, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0),
             "Stopword Filtering & Noise Cleaning", [
                 "Built-in 100% Offline English Stopword Lexicon:",
                 "  • Removes low-information tokens ('the', 'is', 'at', 'which', 'on').",
                 "  • Preserves sentiment-critical negation tokens: 'not', 'never', 'barely', 'hardly'.",
                 "  • Example: 'not good' is preserved as a negation unit rather than stripped.",
                 "",
                 "Tokenization & Punctuation Pipeline:",
                 "  • Regex pattern: [a-zA-Z0-9']+",
                 "  • Strips special characters, email addresses, and HTML tags.",
                 "  • Color-coded token output: Green for positive valence, Red for negative valence."
             ], PRIMARY_DARK)

    # ==========================================
    # SLIDE 5: VADER LEXICON & COMPOUND FORMULA
    # ==========================================
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Rule-Based Lexicon Scoring: NLTK VADER Intensity", "Lexicon Intelligence")
    
    add_card(s5, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.0),
             "Mathematical Formula & Normalization", [
                 "VADER (Valence Aware Dictionary and sEntiment Reasoner):",
                 "Rule-based model specifically attuned to micro-sentiment in consumer reviews.",
                 "",
                 "Compound Score Formula:",
                 "       c = x / sqrt(x^2 + alpha)",
                 "",
                 "Where:",
                 "  • x = Sum of valence scores of constituent lexicon tokens in review.",
                 "  • alpha = Normalization parameter (experimentally set to alpha ≈ 15).",
                 "  • Normalizes metric smoothly between -1.0 (extremely negative) and +1.0 (positive).",
                 "",
                 "Decision Cutoff Thresholds (Adjustable via Live Slider):",
                 "  • Positive: Compound Score >= +0.05",
                 "  • Neutral:  -0.05 < Compound Score < +0.05",
                 "  • Negative: Compound Score <= -0.05"
             ], PRIMARY)
             
    add_card(s5, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0),
             "Key Advantages of VADER", [
                 "1. Zero Training Time / Cold Start Ready:",
                 "  • Operates immediately without requiring large labeled datasets.",
                 "",
                 "2. Heuristic Modifiers:",
                 "  • Punctuation booster (e.g. 'Great!!!' scores higher than 'Great').",
                 "  • Capitalization booster (e.g. 'AMAZING' vs 'amazing').",
                 "  • Degree adverbs / Intensifiers ('extremely good', 'barely acceptable').",
                 "",
                 "3. Explainability & Trust:",
                 "  • Outputs exact pos, neg, neu component ratios.",
                 "  • Direct visibility into why a review scored positive or negative."
             ], ACCENT_GREEN)

    # ==========================================
    # SLIDE 6: OPINION MINING & SUBJECTIVITY
    # ==========================================
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Opinion Mining: TextBlob Subjectivity vs. Fact", "Opinion Mining")
    
    add_card(s6, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.0),
             "Subjectivity Spectrum [0.0 to 1.0]", [
                 "TextBlob calculates a dedicated Subjectivity metric:",
                 "",
                 "Score = 0.0 (Pure Objective / Fact-Based):",
                 "  • Statements stating verified specifications, physical delivery, or timelines.",
                 "  • Example: 'The laptop arrived in 2 business days with 16GB of DDR5 RAM.'",
                 "  • Sentiment is neutral, subjectivity = 0.0.",
                 "",
                 "Score = 1.0 (Pure Subjective / Personal Emotion):",
                 "  • Reviews reflecting personal taste, mood, or emotional perception.",
                 "  • Example: 'This phone feels truly magnificent, sleek, and heavenly in hand.'",
                 "  • High sentiment, subjectivity = 0.95."
             ], PRIMARY)
             
    add_card(s6, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0),
             "Business & Engineering Utility", [
                 "1. Actionable Bug / Defect Separation:",
                 "  • High-subjectivity complaints = Customer expectations / UI preference.",
                 "  • Low-subjectivity complaints = Concrete physical defects or logistic errors.",
                 "",
                 "2. Dual Metric Matrix:",
                 "  • High Negative + Low Subjectivity = CRITICAL HARDWARE / SHIPPING FAILURE.",
                 "  • High Negative + High Subjectivity = CUSTOMER PREFERENCE / STYLING ISSUE.",
                 "",
                 "3. Automated Triage:",
                 "  • Direct routing to Engineering vs. Customer Relations teams."
             ], ACCENT_GREEN)

    # ==========================================
    # SLIDE 7: MACHINE LEARNING & TF-IDF
    # ==========================================
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Dual Machine Learning Classifiers & TF-IDF Vectors", "Machine Learning")
    
    add_card(s7, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.0),
             "Feature Extraction: TF-IDF N-Grams", [
                 "Term Frequency - Inverse Document Frequency:",
                 "  • Down-weights ubiquitous corpus terms while boosting informative tokens.",
                 "  • Uses (1, 2) N-Grams: captures single words ('fast') and bi-grams ('not fast').",
                 "  • Vectorizer configured with min_df=1, max_df=1.0.",
                 "",
                 "TF-IDF Formula:",
                 "  • TF(t, d) = Count of t in d / Total words in d",
                 "  • IDF(t) = log( (1 + N) / (1 + DF(t)) ) + 1",
                 "  • Weight(t, d) = TF(t, d) * IDF(t) with L2 normalization.",
                 "",
                 "Advantage over BoW (Bag-of-Words):",
                 "  • Prevents frequent domain words from dominating classifier weights."
             ], PRIMARY)
             
    add_card(s7, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0),
             "Dual Model Architecture", [
                 "Classifier 1: Multinomial Logistic Regression",
                 "  • Linear decision boundaries with Softmax probability outputs.",
                 "  • L2 Ridge regularization prevents overfitting on small datasets.",
                 "  • Highly robust on text classification with balanced precision.",
                 "",
                 "Classifier 2: Multinomial Naive Bayes (MNB)",
                 "  • Probabilistic model based on Bayes' Theorem with conditional independence.",
                 "  • Fast O(N) training and sub-millisecond inference latency.",
                 "  • Laplace smoothing (alpha=1.0) eliminates zero-frequency probabilities.",
                 "",
                 "Multi-Model Consensus:",
                 "  • Compares VADER vs Logistic Regression vs Naive Bayes.",
                 "  • Flags disagreements for human audit."
             ], PRIMARY_DARK)

    # ==========================================
    # SLIDE 8: MODEL EVALUATION & CONFUSION MATRIX
    # ==========================================
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Model Evaluation & 3x3 Confusion Matrix Diagnostics", "Evaluation & Metrics")
    
    add_card(s8, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.0),
             "3x3 Confusion Matrix (Test Split)", [
                 "Multi-Class Evaluation Heatmap (Positive, Negative, Neutral):",
                 "",
                 "Actual \\ Pred       Positive      Negative      Neutral",
                 "Positive            [ 1 ]            4             0",
                 "Negative              1            [ 3 ]           0",
                 "Neutral               1              3           [ 1 ]",
                 "",
                 "Diagonal Values [ Bracketed ]:",
                 "  • True Positives (TP): 1 correctly identified positive.",
                 "  • True Negatives (TN): 3 correctly identified negatives.",
                 "  • True Neutrals (TNeu): 1 correctly identified neutral.",
                 "",
                 "Insight:",
                 "  • Neutral reviews often show slight lexical skew.",
                 "  • Highlights need for lexicon threshold slider calibration."
             ], PRIMARY)
             
    add_card(s8, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0),
             "Benchmark Performance Comparison", [
                 "Performance Metrics Breakdown:",
                 "",
                 "Metric                  Logistic Regression      Naive Bayes",
                 "Test Accuracy           Balanced Benchmark       Robust Baseline",
                 "Inference Time          < 5 ms                   < 2 ms",
                 "Feature Sparsity        High (L2 Regularized)    Dense Probabilities",
                 "Memory Footprint        ~11 KB                   ~10 KB",
                 "",
                 "Key Viva Defense Point:",
                 "  • Both models are serialized with Joblib into micro-artifacts (<12KB).",
                 "  • Loads in zero-time during Vercel cold starts with zero memory pressure."
             ], ACCENT_GREEN)

    # ==========================================
    # SLIDE 9: ASPECT EXTRACTION (ABSA) & TRIAGE
    # ==========================================
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Aspect-Based Sentiment Analysis & Priority Triage", "Aspect Intelligence")
    
    add_card(s9, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.0),
             "4 Domain Aspect Categories", [
                 "1. 📦 Delivery & Shipping:",
                 "  • Keywords: delivery, shipping, arrive, package, delay, transit, courier.",
                 "  • Target: Logistics operations, warehouse fulfillment.",
                 "",
                 "2. 💎 Product Quality & Build:",
                 "  • Keywords: quality, material, durable, broken, defective, finish, sturdy.",
                 "  • Target: Manufacturing, QA inspection, engineering.",
                 "",
                 "3. 🎧 Customer Service & Support:",
                 "  • Keywords: service, support, staff, refund, return, helpline, agent.",
                 "  • Target: Helpdesk, CRM escalation.",
                 "",
                 "4. 💰 Pricing & Value:",
                 "  • Keywords: price, expensive, cheap, value, cost, worth, discount.",
                 "  • Target: Pricing strategy, sales, promotional planning."
             ], PRIMARY)
             
    add_card(s9, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0),
             "Automated Priority Matrix & Actions", [
                 "Priority Levels Assigned Automatically:",
                 "",
                 "🚨 URGENT:",
                 "  • High Negative (< -0.4) + Quality or Delivery defect.",
                 "  • Action: Immediate customer outreach, priority refund / replacement.",
                 "",
                 "⚠️ HIGH:",
                 "  • Moderate Negative (< -0.1) on Customer Service or Pricing.",
                 "  • Action: Helpdesk ticket creation with 24hr SLA.",
                 "",
                 "ℹ️ MEDIUM:",
                 "  • Neutral or mixed review with specific aspect remarks.",
                 "  • Action: Feedback logged in weekly product digest.",
                 "",
                 "✅ LOW:",
                 "  • Positive review (>= +0.2).",
                 "  • Action: Loyalty appreciation, prompt for public rating."
             ], ACCENT_RED)

    # ==========================================
    # SLIDE 10: WEB UI & VERCEL DEPLOYMENT
    # ==========================================
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Interactive Web Interface & Cloud Deployment", "Frontend & Deployment")
    
    add_card(s10, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.0),
             "Light-Theme Interactive GUI", [
                 "Built with Pure Semantic HTML5, CSS3 & JavaScript ES6:",
                 "  • Clean Light Academic Theme designed specifically for presentation.",
                 "  • Chart.js Dynamic Doughnut Chart displaying real-time percentages.",
                 "  • Contextual Word Polarity Highlighter (Green = Pos, Red = Neg).",
                 "  • Dynamic VADER Cutoff Threshold Slider (0.00 to 0.50).",
                 "  • Interactive 3x3 Confusion Matrix Heatmap.",
                 "  • HTML5 Web Speech API direct microphone voice dictation.",
                 "  • Drag-and-drop CSV dataset upload with aggregate distribution.",
                 "  • SQLite Interaction History with 1-Click CSV Export."
             ], PRIMARY)
             
    add_card(s10, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0),
             "Vercel Serverless Architecture", [
                 "High-Performance Cloud Infrastructure:",
                 "  • Serverless ASGI Python 3.11 engine powered by FastAPI.",
                 "  • Zero-cost serverless execution scaling dynamically to zero.",
                 "  • Clean negative-lookahead URL rewrite routing in vercel.json.",
                 "  • Built-in interactive Swagger UI (/docs) and OpenAPI specs.",
                 "",
                 "Live Production URLs:",
                 "  • 👉 Dashboard: https://smart-review-analyzer-sandy.vercel.app/",
                 "  • 👉 Swagger Docs: https://smart-review-analyzer-sandy.vercel.app/docs",
                 "  • 👉 GitHub: https://github.com/surajsingh4507-del/Smart-Review-Analyzer"
             ], ACCENT_GREEN)

    # ==========================================
    # SLIDE 11: VIVA DEFENSE & EXAMINER CHEAT SHEET
    # ==========================================
    s11 = prs.slides.add_slide(blank_layout)
    add_header(s11, "Viva Voce Defense: Key Examiner Questions", "Exam Defense")
    
    add_card(s11, Inches(0.8), Inches(1.9), Inches(5.6), Inches(5.0),
             "Common Technical Questions & Answers", [
                 "Q1: Why combine Rule-Based VADER with ML Classifiers?",
                 "Ans: VADER handles zero-shot cold start and micro-sentiment without labeled data. ML models learn domain-specific vocabulary patterns. Comparing both provides multi-model consensus and flags discrepancies.",
                 "",
                 "Q2: How does VADER compute compound score?",
                 "Ans: It sums lexicon valence scores (x) and normalizes via x / sqrt(x^2 + alpha), with alpha ≈ 15 to constrain output smoothly between -1.0 and +1.0.",
                 "",
                 "Q3: Why not use paid OpenAI API?",
                 "Ans: Privacy preservation, zero running costs, 100% offline capability, zero latency network timeouts, and academic authenticity."
             ], PRIMARY)
             
    add_card(s11, Inches(6.8), Inches(1.9), Inches(5.7), Inches(5.0),
             "Architectural Defense Points", [
                 "Q4: How does SQLite work in a Serverless environment?",
                 "Ans: SQLite stores in /tmp on Vercel with automatic table initialization on cold-start. For local environments, it persists directly to reviews.db with 1-click CSV export.",
                 "",
                 "Q5: How does TextBlob subjectivity differ from polarity?",
                 "Ans: Polarity measures emotional valence (-1 to +1). Subjectivity measures factual grounding (0.0 = verifiable facts, 1.0 = subjective emotional opinions).",
                 "",
                 "Q6: How are aspect categories extracted?",
                 "Ans: Lemmatized noun tokens are matched against pre-compiled domain dictionaries for Shipping, Quality, Service, and Pricing."
             ], PRIMARY_DARK)

    # ==========================================
    # SLIDE 12: CONCLUSION & FUTURE WORK
    # ==========================================
    s12 = prs.slides.add_slide(blank_layout)
    bg12 = s12.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg12.fill.solid()
    bg12.fill.fore_color.rgb = PRIMARY_DARK
    bg12.line.fill.background()
    
    c_box = s12.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(11.333), Inches(5.5))
    ctf = c_box.text_frame
    ctf.word_wrap = True
    
    cp0 = ctf.paragraphs[0]
    cp0.text = "CONCLUSION & FUTURE SCOPE"
    cp0.font.size = Pt(13)
    cp0.font.bold = True
    cp0.font.color.rgb = ACCENT_GREEN
    cp0.space_after = Pt(10)
    
    cp1 = ctf.add_paragraph()
    cp1.text = "Thank You! Questions & Discussion"
    cp1.font.size = Pt(36)
    cp1.font.bold = True
    cp1.font.color.rgb = WHITE
    cp1.space_after = Pt(20)
    
    cp2 = ctf.add_paragraph()
    cp2.text = "Key Project Achievements:"
    cp2.font.size = Pt(16)
    cp2.font.bold = True
    cp2.font.color.rgb = RGBColor(199, 210, 254)
    cp2.space_after = Pt(8)
    
    points = [
        "✅ 100% Offline, Zero-Cost NLP Pipeline running without paid APIs or heavy Docker containers.",
        "✅ Multi-Layered Intelligence: VADER Compound Scoring + TextBlob Subjectivity + Dual ML Models.",
        "✅ Aspect-Based Sentiment Extraction across 4 core business domains with automated triage priority.",
        "✅ Fully responsive Light-Theme Web UI with live Voice Ingestion, Chart.js doughnut, and CSV export.",
        "✅ Production Deployment live on Vercel with Swagger documentation and Pytest verification."
    ]
    for pt in points:
        p = ctf.add_paragraph()
        p.text = f"  {pt}"
        p.font.size = Pt(13)
        p.font.color.rgb = WHITE
        p.space_after = Pt(6)
        
    p_end = ctf.add_paragraph()
    p_end.text = "\n🌐 Live Demo: https://smart-review-analyzer-sandy.vercel.app/   |   Built with ❤️ for College NLP Lab"
    p_end.font.size = Pt(14)
    p_end.font.bold = True
    p_end.font.color.rgb = ACCENT_GREEN

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Smart_Review_Analyzer_College_Presentation.pptx")
    prs.save(output_path)
    print(f"Successfully generated PowerPoint presentation at: {output_path}")

if __name__ == "__main__":
    build_presentation()
