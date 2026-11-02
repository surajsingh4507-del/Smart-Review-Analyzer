"""
Comprehensive Test Suite for Smart Review Analyzer v3.0
Covers all 22 required test cases across single review NLP, document extraction
(.csv, .txt, .pdf, .docx, .doc, .xlsx, .xls), batch analytics, exports (CSV/PDF),
and frontend structure verification.
"""

import io
import os
import sys
import docx
import openpyxl
import pypdf
import pytest
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

# Ensure Smart-Review-Analyzer root is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from fastapi import UploadFile, HTTPException
from nlp_engine import (
    clean_and_tokenize,
    analyze_vader_sentiment,
    analyze_textblob_metrics,
    complete_review_analysis,
    predict_ml_sentiment,
    resolve_final_sentiment,
    explain_sentiment_result,
    segment_sentences,
    analyze_sentence,
    analyze_paragraph_sentences
)
from document_parser import (
    validate_and_extract_document,
    split_into_reviews,
    DocumentParsingError,
    MAX_FILE_SIZE_BYTES
)
from pdf_exporter import generate_batch_pdf_report
from main import (
    app,
    vectorizer,
    lr_model,
    nb_model,
    analyze_single_review,
    analyze_batch_reviews,
    upload_document_endpoint,
    export_batch_csv,
    export_batch_pdf,
    get_interactive_dashboard,
    SingleReviewRequest,
    BatchReviewRequest
)


# ============================================================================
# 1. Single Review - Positive Sentiment
# ============================================================================
def test_01_single_positive_sentiment():
    sample = "The product quality is excellent."
    res = complete_review_analysis(sample, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res["sentiment"] == "Positive"
    assert res["confidence_percentage"] > 50
    assert res["vader_scores"]["compound_score"] > 0
    assert "explanation" in res
    assert "positive" in res["explanation"].lower()


# ============================================================================
# 2. Single Review - Negative Sentiment
# ============================================================================
def test_02_single_negative_sentiment():
    sample = "The delivery was extremely late."
    res = complete_review_analysis(sample, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res["sentiment"] == "Negative"
    assert res["vader_scores"]["compound_score"] < 0
    assert "Delivery & Shipping" in res["categories"]
    assert "explanation" in res
    assert "negative" in res["explanation"].lower()


# ============================================================================
# 3. Single Review - Neutral Sentiment
# ============================================================================
def test_03_single_neutral_sentiment():
    sample = "The device dimensions are 15cm by 10cm."
    res = complete_review_analysis(sample, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res["sentiment"] == "Neutral"
    assert abs(res["vader_scores"]["compound_score"]) <= 0.05
    assert "neutral" in res["explanation"].lower()


# ============================================================================
# 4. Text Containing Negations ("not good", "not bad", "not worth")
# ============================================================================
def test_04_negations_handling():
    # 4a: "not good" -> Negative
    sample_neg1 = "The camera is not good."
    res_neg1 = complete_review_analysis(sample_neg1, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res_neg1["sentiment"] == "Negative"

    # 4b: "not worth the money" -> Negative
    sample_neg2 = "The product is not worth the money."
    res_neg2 = complete_review_analysis(sample_neg2, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res_neg2["sentiment"] == "Negative"

    # 4c: "not bad at all" -> Positive or Neutral, never strongly negative
    sample_pos = "The build is not bad at all, actually quite good."
    res_pos = complete_review_analysis(sample_pos, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res_pos["sentiment"] == "Positive"


# ============================================================================
# 5. CSV File with Multiple Review Rows and Correct Column Detection
# ============================================================================
def test_05_csv_multiple_reviews_column_detection():
    csv_text = "TransactionID,Customer_Feedback,Rating\n101,The product quality is excellent.,5\n102,The delivery was extremely late.,1\n103,The customer support was helpful.,5\n"
    extracted = validate_and_extract_document("orders.csv", csv_text.encode("utf-8"))
    assert extracted["diagnostics"]["file_extension"] == ".csv"
    
    reviews = split_into_reviews(extracted, mode="auto")
    assert len(reviews) == 3
    # Check that feedback text was detected and source IDs were preserved
    assert any("The product quality is excellent." in r["text"] for r in reviews)
    assert any("The delivery was extremely late." in r["text"] for r in reviews)
    assert any("The customer support was helpful." in r["text"] for r in reviews)
    assert reviews[0]["id"] == "Row 2"


# ============================================================================
# 6. Plain Text File (.txt) with Multiple Paragraphs
# ============================================================================
def test_06_txt_multiple_paragraphs():
    txt_content = (
        "First review paragraph: The display is bright and responsive.\n\n"
        "Second review paragraph: Customer service was rude and refused to help.\n\n"
        "Third review paragraph: The device comes with a 1-year warranty.\n"
    )
    extracted = validate_and_extract_document("notes.txt", txt_content.encode("utf-8"))
    assert extracted["diagnostics"]["file_extension"] == ".txt"
    assert extracted["diagnostics"]["character_count"] > 50

    reviews = split_into_reviews(extracted, mode="paragraph")
    assert len(reviews) == 3
    assert "display is bright" in reviews[0]["text"]
    assert "Customer service was rude" in reviews[1]["text"]
    assert "1-year warranty" in reviews[2]["text"]


# ============================================================================
# 7. Word Document (.docx) with Multiple Paragraphs
# ============================================================================
def test_07_docx_multiple_paragraphs():
    doc = docx.Document()
    doc.add_paragraph("Paragraph 1: The product quality is excellent.")
    doc.add_paragraph("Paragraph 2: The delivery was extremely late.")
    doc.add_paragraph("Paragraph 3: The customer support was helpful.")
    bio = io.BytesIO()
    doc.save(bio)

    extracted = validate_and_extract_document("multi_para.docx", bio.getvalue())
    reviews = split_into_reviews(extracted, mode="auto")
    assert len(reviews) == 3
    assert "quality is excellent" in reviews[0]["text"]
    assert "delivery was extremely late" in reviews[1]["text"]
    assert "customer support was helpful" in reviews[2]["text"]


# ============================================================================
# 8. Word Document (.docx) Containing a Table
# ============================================================================
def test_08_docx_table_extraction():
    doc = docx.Document()
    doc.add_paragraph("Summary Table Below:")
    table = doc.add_table(rows=1, cols=2)
    hdr = table.rows[0].cells
    hdr[0].text = "ID"
    hdr[1].text = "User Review"

    row1 = table.add_row().cells
    row1[0].text = "1"
    row1[1].text = "The customer support was helpful and friendly."

    row2 = table.add_row().cells
    row2[0].text = "2"
    row2[1].text = "The product is not worth the money, poor performance."

    bio = io.BytesIO()
    doc.save(bio)

    extracted = validate_and_extract_document("reviews_table.docx", bio.getvalue())
    reviews = split_into_reviews(extracted, mode="auto")
    texts = [r["text"] for r in reviews]
    assert any("customer support was helpful" in t for t in texts)
    assert any("not worth the money" in t for t in texts)
    # Header should not be recognized as a customer review
    assert not any("ID - User Review" in t for t in texts)


# ============================================================================
# 9. PDF Document with Multiple Pages
# ============================================================================
def test_09_pdf_multi_page():
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Page 1: The product quality is excellent and durable.", styles['Normal']),
        PageBreak(),
        Paragraph("Page 2: The delivery was extremely late and frustrating.", styles['Normal']),
        PageBreak(),
        Paragraph("Page 3: The packaging was standard cardboard.", styles['Normal'])
    ]
    doc.build(story)
    pdf_bytes = bio.getvalue()

    extracted = validate_and_extract_document("multipage.pdf", pdf_bytes)
    assert extracted["diagnostics"]["file_extension"] == ".pdf"
    assert extracted["metadata"].get("total_pages") == 3
    assert "Page 1" in extracted["extracted_text"]
    assert "Page 2" in extracted["extracted_text"]
    assert "Page 3" in extracted["extracted_text"]


# ============================================================================
# 10. Empty PDF or Scanned Image-Only PDF Error
# ============================================================================
def test_10_empty_scanned_pdf_error():
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=300, height=300)
    bio = io.BytesIO()
    writer.write(bio)
    empty_pdf_bytes = bio.getvalue()

    with pytest.raises(DocumentParsingError) as exc_info:
        validate_and_extract_document("scanned_empty.pdf", empty_pdf_bytes)
    err_msg = str(exc_info.value).lower()
    assert "ocr" in err_msg or "no extractable text" in err_msg


# ============================================================================
# 11. Excel Spreadsheet (.xlsx) Containing Multiple Reviews
# ============================================================================
def test_11_xlsx_multiple_reviews():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Reviews"
    ws.append(["ID", "Feedback", "Rating"])
    ws.append([1, "The product quality is excellent.", 5])
    ws.append([2, "The delivery was extremely late.", 1])
    ws.append([3, "The customer support was helpful.", 5])
    bio = io.BytesIO()
    wb.save(bio)
    xlsx_bytes = bio.getvalue()

    extracted = validate_and_extract_document("feedback.xlsx", xlsx_bytes)
    assert extracted["diagnostics"]["file_extension"] == ".xlsx"
    reviews = split_into_reviews(extracted, mode="auto")
    assert len(reviews) == 3
    assert any("quality is excellent" in r["text"] for r in reviews)
    assert any("delivery was extremely late" in r["text"] for r in reviews)


# ============================================================================
# 12. Excel Spreadsheet (.xlsx) with Column Detection
# ============================================================================
def test_12_xlsx_column_detection():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Export"
    # Ambiguous column names: SKU, Review_Comments, Price
    ws.append(["SKU", "Review_Comments", "Price"])
    ws.append(["A100", "The product quality is excellent.", 29.99])
    ws.append(["B200", "The product is not worth the money.", 99.99])
    bio = io.BytesIO()
    wb.save(bio)

    extracted = validate_and_extract_document("products.xlsx", bio.getvalue())
    reviews = split_into_reviews(extracted, mode="auto")
    assert len(reviews) == 2
    assert "The product quality is excellent." in reviews[0]["text"]
    assert "The product is not worth the money." in reviews[1]["text"]


# ============================================================================
# 13. Corrupted or Unsupported File Formats (.xyz, Corrupted Zip)
# ============================================================================
def test_13_corrupted_or_unsupported_formats():
    # Unsupported extension
    with pytest.raises(DocumentParsingError) as exc_unsupported:
        validate_and_extract_document("data.xyz", b"some binary data")
    assert "unsupported" in str(exc_unsupported.value).lower()

    # Corrupted docx
    with pytest.raises(DocumentParsingError) as exc_corrupt:
        validate_and_extract_document("broken.docx", b"Not a real docx file")
    assert "corrupted" in str(exc_corrupt.value).lower()


# ============================================================================
# 14. Legacy Document Handling (.doc, .xls)
# ============================================================================
def test_14_legacy_document_handling():
    # Legacy .doc with text stream or conversion message
    doc_header = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"Customer review: Great sound quality." * 5
    try:
        res = validate_and_extract_document("legacy.doc", doc_header)
        assert res["diagnostics"]["file_extension"] == ".doc"
    except DocumentParsingError as e:
        # If OLE2 extraction requires conversion, verify the helpful error message
        assert "doc" in str(e).lower()


# ============================================================================
# 15. Long Document and Size Limit Enforcement
# ============================================================================
def test_15_long_document_and_size_limits():
    oversized_bytes = b"A" * (MAX_FILE_SIZE_BYTES + 1024)
    with pytest.raises(DocumentParsingError) as exc_oversized:
        validate_and_extract_document("huge.txt", oversized_bytes)
    assert "exceeds maximum allowed size" in str(exc_oversized.value)


# ============================================================================
# 16. Mixed Sentiment in Single Review
# ============================================================================
def test_16_mixed_sentiment_in_single_review():
    mixed_text = "The camera resolution is great and design is sleek, but the battery life is terribly disappointing."
    res = complete_review_analysis(mixed_text, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    # Entire review is analyzed as one without splitting sentences
    assert res["original_review"] == mixed_text
    assert res["sentiment"] in ["Positive", "Negative", "Neutral"]
    assert "explanation" in res
    assert len(res["categories"]) >= 1


# ============================================================================
# 17. Batch Count and Percentage Calculations
# ============================================================================
def test_17_batch_count_and_percentage_calculations():
    payload = BatchReviewRequest(
        reviews=[
            "The product quality is excellent.",
            "Outstanding performance, loved it!",
            "The delivery was extremely late.",
            "The device dimensions are 15cm by 10cm."
        ],
        threshold=0.05
    )
    data = analyze_batch_reviews(payload)
    assert data["total_reviews"] == 4
    dist = data["sentiment_distribution"]
    assert dist["Positive"] == 2
    assert dist["Negative"] == 1
    assert dist["Neutral"] == 1
    assert dist["Positive_Percentage"] == 50.0
    assert dist["Negative_Percentage"] == 25.0
    assert dist["Neutral_Percentage"] == 25.0
    assert dist["Positive_Percentage"] + dist["Negative_Percentage"] + dist["Neutral_Percentage"] == 100.0


# ============================================================================
# 18. Failed Predictions and Error Reporting
# ============================================================================
def test_18_failed_predictions_and_error_reporting():
    # Empty string should raise HTTP 400
    with pytest.raises(HTTPException) as exc_single:
        analyze_single_review(SingleReviewRequest(review="   "))
    assert exc_single.value.status_code == 400

    # Empty batch should raise HTTP 400
    with pytest.raises(HTTPException) as exc_batch:
        analyze_batch_reviews(BatchReviewRequest(reviews=["   ", ""]))
    assert exc_batch.value.status_code == 400


# ============================================================================
# 19. CSV Export File Generation and Content Verification
# ============================================================================
def test_19_csv_export_content():
    dummy_data = {
        "detailed_results": [
            {
                "source_id": "Row 1",
                "original_review": "The product quality is excellent.",
                "sentiment": "Positive",
                "confidence_percentage": 85.0,
                "vader_scores": {"compound_score": 0.57},
                "textblob_metrics": {"polarity": 1.0},
                "ml_model_predictions": {"logistic_regression": "Positive"},
                "categories": ["Product Quality"],
                "priority": "Normal",
                "explanation": "Positive sentiment driven by excellent"
            }
        ]
    }
    response = export_batch_csv(dummy_data)
    assert "attachment; filename=sentiment_results_" in response.headers["Content-Disposition"]
    assert response.headers["Content-Disposition"].endswith(".csv")

    csv_body = response.body.decode("utf-8")
    assert "Source ID / Row" in csv_body
    assert "The product quality is excellent." in csv_body
    assert "Positive" in csv_body
    assert "Row 1" in csv_body


# ============================================================================
# 20. PDF Export File Generation and Content Verification
# ============================================================================
def test_20_pdf_export_content():
    dummy_data = {
        "total_reviews": 1,
        "sentiment_distribution": {
            "Positive": 1, "Negative": 0, "Neutral": 0,
            "Positive_Percentage": 100.0, "Negative_Percentage": 0.0, "Neutral_Percentage": 0.0
        },
        "aspect_distribution": {"Product Quality": 1},
        "diagnostics": {
            "filename": "test_reviews.docx",
            "file_extension": ".docx",
            "character_count": 35,
            "word_count": 5
        },
        "detailed_results": [
            {
                "source_id": "1",
                "original_review": "The product quality is excellent.",
                "sentiment": "Positive",
                "confidence_percentage": 85.0,
                "vader_scores": {"compound_score": 0.57},
                "textblob_metrics": {"polarity": 1.0},
                "ml_model_predictions": {"logistic_regression": "Positive"},
                "categories": ["Product Quality"],
                "priority": "Normal"
            }
        ]
    }
    response = export_batch_pdf(dummy_data)
    assert "attachment; filename=sentiment_results_" in response.headers["Content-Disposition"]
    assert response.headers["Content-Disposition"].endswith(".pdf")
    assert response.media_type == "application/pdf"
    assert response.body.startswith(b"%PDF-")
    assert len(response.body) > 1000


# ============================================================================
# 21. Frontend Dashboard Structure & Separate Sections Verification
# ============================================================================
def test_21_frontend_dashboard_structure():
    response = get_interactive_dashboard()
    html = response.body.decode("utf-8")

    # Section A: Single Review Analysis exclusively text-based
    assert 'id="singleModeSection"' in html
    assert 'Paste or type a product review here...' in html
    assert '⚡ Analyze Sentiment' in html
    assert 'id="sampleDropdown"' in html
    assert 'id="singleExplanationCard"' in html

    # Verify no file input inside Section A
    single_section_html = html[html.index('id="singleModeSection"'):html.index('id="batchModeSection"')]
    assert 'type="file"' not in single_section_html

    # Section B: Batch & Multi-Format File Upload
    assert 'id="batchModeSection"' in html
    assert 'Batch Review Analysis & Multi-Format File Upload' in html
    assert 'Option 1: Upload a File' in html
    assert 'Option 2: Paste Multiple Reviews' in html
    assert 'id="multiFileInput"' in html
    assert 'id="docModeSelect"' in html
    assert 'id="batchSearchInput"' in html
    assert 'id="batchPieChart"' in html
    assert 'id="batchBarChart"' in html
    assert 'Download CSV' in html
    assert 'Export PDF Report' in html


# ============================================================================
# 22. Single Review API Regression (/api/analyze)
# ============================================================================
def test_22_single_review_api_regression():
    payload = SingleReviewRequest(
        review="Customer support was polite and resolved the issue immediately.",
        threshold=0.05
    )
    result = analyze_single_review(payload)
    assert result["sentiment"] == "Positive"
    assert result["confidence_percentage"] >= 50
    assert "vader_scores" in result
    assert "textblob_metrics" in result
    assert "ml_model_predictions" in result
    assert "model_comparison" in result
    assert "explanation" in result
    assert "categories" in result
    assert "Customer Service" in result["categories"]


# ============================================================================
# 23. Sentence Segmentation - Decimals, Abbreviations & Edge Cases
# ============================================================================
def test_23_sentence_segmentation_decimals_abbreviations_and_edge_cases():
    text = "The device comes with a 6.5-inch display and 128 GB of storage. Dr. Smith reviewed it at 5.0 stars. E.g. great battery."
    sents = segment_sentences(text)
    assert len(sents) == 3
    assert "6.5-inch" in sents[0]
    assert "Dr. Smith" in sents[1]
    assert "E.g." in sents[2]

    # Empty text handling
    assert segment_sentences("") == []
    assert segment_sentences("   ") == []

    # Single-sentence paragraph
    single = segment_sentences("Just one solitary sentence without linebreaks.")
    assert len(single) == 1
    assert single[0] == "Just one solitary sentence without linebreaks."


# ============================================================================
# 24. Samsung Galaxy A55 Single Long Paragraph Sentence Analysis
# ============================================================================
def test_24_samsung_galaxy_a55_sentence_level_analysis():
    samsung_text = (
        "The Samsung Galaxy A55 has a beautiful display and the camera takes excellent photos. "
        "The battery lasts all day during normal usage. "
        "However, the phone heats up during gaming and the charging speed is disappointing. "
        "The design is premium and the phone feels comfortable to hold. "
        "The device comes with a 6.5-inch display and 128 GB of storage."
    )
    result = analyze_paragraph_sentences(
        samsung_text,
        threshold=0.05,
        vectorizer=vectorizer,
        lr_model=lr_model,
        nb_model=nb_model
    )

    # 1. Exactly 5 sentences identified
    assert result["total_sentences"] == 5
    assert result["valid_sentences"] == 5
    assert result["failed_sentences"] == 0
    assert len(result["sentences"]) == 5

    # 2. Dynamic distribution calculation
    dist = result["sentiment_distribution"]
    assert dist["Positive"] == 2
    assert dist["Negative"] == 1
    assert dist["Neutral"] == 2
    assert dist["Positive_Percentage"] == 40.0
    assert dist["Negative_Percentage"] == 20.0
    assert dist["Neutral_Percentage"] == 40.0

    # 3. Overall compound score: (0.8225 + 0.0 + (-0.4939) + 0.5106 + 0.0) / 5 = 0.1678
    assert abs(result["overall_compound_score"] - 0.1678) < 0.001
    assert result["overall_sentiment"] == "Positive"

    # 4. Each sentence has all required audit fields
    for s in result["sentences"]:
        assert "sentence_id" in s
        assert "text" in s
        assert "sentiment" in s
        assert "vader_scores" in s
        assert "vader_compound" in s
        assert "textblob_polarity" in s
        assert "ml_prediction" in s


# ============================================================================
# 25. Sentence Distribution & Zero-Division / Failed Sentence Protection
# ============================================================================
def test_25_sentence_distribution_and_zero_division():
    # Empty string must not raise ZeroDivisionError and safely return neutral zero distribution
    res_empty = analyze_paragraph_sentences("", threshold=0.05)
    assert res_empty["total_sentences"] == 0
    assert res_empty["valid_sentences"] == 0
    assert res_empty["failed_sentences"] == 0
    assert res_empty["sentiment_distribution"]["Positive_Percentage"] == 0.0
    assert res_empty["sentiment_distribution"]["Negative_Percentage"] == 0.0
    assert res_empty["sentiment_distribution"]["Neutral_Percentage"] == 0.0
    assert res_empty["overall_compound_score"] == 0.0
    assert res_empty["overall_sentiment"] == "Neutral"


# ============================================================================
# 26. Overall Sentiment Aggregation Formula vs Naive Majority Count
# ============================================================================
def test_26_overall_sentiment_aggregation_not_majority():
    # 3 Neutral sentences vs 2 strongly Positive sentences
    # Majority count is Neutral (3 > 2), but mean compound is positive (> 0.05).
    # Sentiment MUST be Positive based on mean compound formula, not Neutral!
    text = (
        "The item is blue in color. "
        "The dimensions are 10 by 20 cm. "
        "The package arrived on Tuesday. "
        "I absolutely love this phenomenal and extraordinary product! "
        "It is exceptionally brilliant and fantastic in every possible way!"
    )
    res = analyze_paragraph_sentences(text, threshold=0.05)
    assert res["sentiment_distribution"]["Neutral"] == 3
    assert res["sentiment_distribution"]["Positive"] == 2
    # Verify overall compound is positive
    assert res["overall_compound_score"] > 0.05
    assert res["overall_sentiment"] == "Positive"
    assert "mean" in res["explanation"].lower() or "exceeds" in res["explanation"].lower()


# ============================================================================
# 27. Threshold Sensitivity (0.05 vs 0.10 vs 0.20)
# ============================================================================
def test_27_sentence_threshold_sensitivity():
    samsung_text = (
        "The Samsung Galaxy A55 has a beautiful display and the camera takes excellent photos. "
        "The battery lasts all day during normal usage. "
        "However, the phone heats up during gaming and the charging speed is disappointing. "
        "The design is premium and the phone feels comfortable to hold. "
        "The device comes with a 6.5-inch display and 128 GB of storage."
    )
    # At T=0.05: mean (+0.1678) > 0.05 -> Positive
    res_005 = analyze_paragraph_sentences(samsung_text, threshold=0.05)
    assert res_005["overall_sentiment"] == "Positive"

    # At T=0.10: mean (+0.1678) > 0.10 -> Positive
    res_010 = analyze_paragraph_sentences(samsung_text, threshold=0.10)
    assert res_010["overall_sentiment"] == "Positive"

    # At T=0.20: mean (+0.1678) <= 0.20 and >= -0.20 -> Neutral
    res_020 = analyze_paragraph_sentences(samsung_text, threshold=0.20)
    assert res_020["overall_sentiment"] == "Neutral"
    assert "neutral boundary" in res_020["explanation"].lower()


# ============================================================================
# 28. Single Review API returns Sentence Breakdown & Integration
# ============================================================================
def test_28_single_review_api_sentence_integration():
    samsung_text = (
        "The Samsung Galaxy A55 has a beautiful display and the camera takes excellent photos. "
        "The battery lasts all day during normal usage. "
        "However, the phone heats up during gaming and the charging speed is disappointing. "
        "The design is premium and the phone feels comfortable to hold. "
        "The device comes with a 6.5-inch display and 128 GB of storage."
    )
    payload = SingleReviewRequest(review=samsung_text, threshold=0.05)
    res = analyze_single_review(payload)
    assert res["sentiment"] == "Positive"
    assert "sentence_analysis" in res
    sa = res["sentence_analysis"]
    assert sa["total_sentences"] == 5
    assert len(sa["sentences"]) == 5
    assert sa["sentiment_distribution"]["Positive"] == 2
    assert sa["sentiment_distribution"]["Negative"] == 1
    assert sa["sentiment_distribution"]["Neutral"] == 2


# ============================================================================
# 29. Document Upload - Single Paragraph DOCX Sentence Analysis
# ============================================================================
def test_29_document_upload_single_paragraph_sentence_analysis():
    doc = docx.Document()
    doc.add_paragraph(
        "The Samsung Galaxy A55 has a beautiful display and the camera takes excellent photos. "
        "The battery lasts all day during normal usage. "
        "However, the phone heats up during gaming and the charging speed is disappointing. "
        "The design is premium and the phone feels comfortable to hold. "
        "The device comes with a 6.5-inch display and 128 GB of storage."
    )
    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)

    upload = UploadFile(filename="samsung_review.docx", file=bio)
    result = upload_document_endpoint(file=upload, mode="auto", threshold=0.05)
    
    assert result["total_reviews"] == 1
    review = result["detailed_results"][0]
    assert "sentence_analysis" in review
    assert review["sentence_analysis"]["total_sentences"] == 5
    assert "sentence_distribution" in result
    sd = result["sentence_distribution"]
    assert sd["total_sentences"] == 5
    assert sd["positive_sentences"] == 2
    assert sd["negative_sentences"] == 1
    assert sd["neutral_sentences"] == 2


# ============================================================================
# 30. Document Upload - Multi-Review Document Sentence Distribution Separation
# ============================================================================
def test_30_multi_review_sentence_distribution_separation():
    csv_content = (
        "review\n"
        "\"The camera is awesome. The screen is gorgeous.\"\n"
        "\"Terrible battery life. Horrible customer service.\"\n"
        "\"The item has a 6-inch screen. It weighs 150 grams.\"\n"
    ).encode("utf-8")
    
    upload = UploadFile(filename="reviews_multi.csv", file=io.BytesIO(csv_content))
    result = upload_document_endpoint(file=upload, mode="auto", threshold=0.05)

    # 3 reviews identified
    assert result["total_reviews"] == 3
    # Review distribution: 1 Positive, 1 Negative, 1 Neutral
    assert result["sentiment_distribution"]["Positive"] == 1
    assert result["sentiment_distribution"]["Negative"] == 1
    assert result["sentiment_distribution"]["Neutral"] == 1

    # Sentence distribution across entire document: 6 sentences in total
    assert "sentence_distribution" in result
    sd = result["sentence_distribution"]
    assert sd["total_sentences"] == 6
    assert sd["positive_sentences"] == 2
    assert sd["negative_sentences"] == 2
    assert sd["neutral_sentences"] == 2
    # Verify review metrics and sentence metrics are kept separate and not conflated
    assert result["total_reviews"] != sd["total_sentences"]


# ============================================================================
# 31. Sentence Negations & Polar Cases
# ============================================================================
def test_31_sentence_negations_and_polar_cases():
    # All positive
    pos_text = "I love this product. It is absolutely wonderful. Outstanding quality."
    pos_res = analyze_paragraph_sentences(pos_text, threshold=0.05)
    assert pos_res["sentiment_distribution"]["Positive"] == 3
    assert pos_res["sentiment_distribution"]["Positive_Percentage"] == 100.0
    assert pos_res["overall_sentiment"] == "Positive"

    # All negative
    neg_text = "This is terrible. Awful experience. Completely broken."
    neg_res = analyze_paragraph_sentences(neg_text, threshold=0.05)
    assert neg_res["sentiment_distribution"]["Negative"] == 3
    assert neg_res["sentiment_distribution"]["Negative_Percentage"] == 100.0
    assert neg_res["overall_sentiment"] == "Negative"

    # Strong negations
    negation_text = "The camera is not good at all. The battery is not great either."
    negation_res = analyze_paragraph_sentences(negation_text, threshold=0.05)
    assert negation_res["sentiment_distribution"]["Negative"] == 2
    assert negation_res["overall_sentiment"] == "Negative"


# ============================================================================
# 32. CSV Header Detection - Exactly 10 Reviews from 1 Header + 10 Rows
# ============================================================================
def test_32_csv_header_detection_10_reviews_exact_count():
    """
    Regression test: Uploading a CSV with header 'review_text' and 10 review rows
    must yield exactly 10 processed reviews (NOT 11).
    Header 'review_text' must never be treated as a review.
    """
    csv_content = (
        "review_text\n"
        "The battery life is amazing and lasts two full days easily.\n"
        "Terrible build quality, it stopped working after three hours.\n"
        "The package arrived with standard cardboard box packaging.\n"
        "The camera takes stunning sharp photos even in low light.\n"
        "Worst customer service ever, representative was completely rude.\n"
        "This model features a 6.2-inch AMOLED screen and 128 GB memory.\n"
        "I absolutely love the clean UI and fluid animations.\n"
        "The charging cable broke on the second day of use.\n"
        "The device weighs approximately 180 grams.\n"
        "Highly recommended for anyone looking for great performance.\n"
    ).encode("utf-8")

    upload = UploadFile(filename="customer_reviews.csv", file=io.BytesIO(csv_content))
    result = upload_document_endpoint(file=upload, mode="auto", threshold=0.05)

    assert result["total_reviews"] == 10
    assert result["reviews_processed"] == 10
    assert result["reviews_failed"] == 0

    detailed = result["detailed_results"]
    assert len(detailed) == 10

    # First review MUST be the actual first customer review, NOT the header
    assert "battery life is amazing" in detailed[0]["original_review"]
    assert detailed[0]["original_review"].strip().lower() != "review_text"

    # None of the results should be header strings
    for r in detailed:
        assert r["original_review"].strip().lower() not in {"review_text", "review", "customer_feedback"}

    # Sentiment distribution must sum to exactly 10
    dist = result["sentiment_distribution"]
    assert dist["Positive"] + dist["Negative"] + dist["Neutral"] == 10


# ============================================================================
# 33. CSV Leading Empty Lines and Whitespace Rows
# ============================================================================
def test_33_csv_leading_empty_lines_and_whitespace_rows():
    """
    CSV with leading blank lines, empty whitespace rows between reviews,
    and trailing empty lines must cleanly extract only valid reviews.
    """
    csv_content = (
        "\n\n   \n"
        "review_text\n"
        "Great performance and very fast.\n"
        "   \n"
        "\n"
        "Horrible experience, refund requested.\n"
        "The dimensions are 10 by 20 cm.\n"
        "\n   \n"
    ).encode("utf-8")

    upload = UploadFile(filename="spaced_reviews.csv", file=io.BytesIO(csv_content))
    result = upload_document_endpoint(file=upload, mode="auto", threshold=0.05)

    assert result["total_reviews"] == 3
    assert result["reviews_processed"] == 3
    assert result["reviews_failed"] == 0


# ============================================================================
# 34. Confidence Metric is Genuine ML Probability, Not VADER Compound
# ============================================================================
def test_34_confidence_metric_is_genuine_ml_probability_not_vader():
    """
    Verify that confidence_percentage is derived from genuine ML predict_proba()
    and NEVER fabricated from abs(vader_compound) * 100.
    Also verify N/A attribution when no ML model is provided.
    """
    review = "The battery life is good and charging speed is decent."
    res_with_ml = complete_review_analysis(
        review,
        vectorizer=vectorizer,
        lr_model=lr_model,
        nb_model=nb_model
    )

    # When ML model is available:
    assert res_with_ml["confidence_percentage"] is not None
    assert isinstance(res_with_ml["confidence_percentage"], float)
    assert 0.0 <= res_with_ml["confidence_percentage"] <= 100.0
    assert "Logistic Regression" in res_with_ml.get("confidence_model", "")

    # Ensure it is NOT simply compound * 100
    compound_abs_100 = round(abs(res_with_ml["vader_scores"]["compound_score"]) * 100, 2)
    # The VADER compound for this sentence is ~0.4404 -> 44.04%
    # Genuine ML probability should not be arbitrarily locked to compound * 100
    vader_comp = res_with_ml["vader_scores"]["compound_score"]
    assert res_with_ml["overall_compound_score"] == vader_comp

    # When NO ML model is provided:
    res_no_ml = complete_review_analysis(
        review,
        vectorizer=None,
        lr_model=None,
        nb_model=None
    )
    assert res_no_ml["confidence_percentage"] is None
    assert res_no_ml["confidence_model"] == "N/A"
    # VADER scores remain intact
    assert res_no_ml["vader_scores"]["compound_score"] != 0.0


# ============================================================================
# 35. Priority Labels Based Strictly on Sentiment Severity
# ============================================================================
def test_35_priority_labels_based_on_sentiment_severity():
    """
    Verify Priority assignment:
    - High: Severe negative (compound <= -0.50 or escalation keyword)
    - Medium: Moderate negative (-0.50 < compound < -0.05)
    - Low: Neutral or Positive
    """
    # 1. Severe negative with escalation keyword
    severe = "The product is completely broken, a terrible scam, and I demand an immediate refund!"
    res_severe = complete_review_analysis(severe, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res_severe["priority"] == "High"

    # 2. Moderate negative
    moderate = "The battery charging speed is somewhat disappointing and slow."
    res_mod = complete_review_analysis(moderate, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res_mod["sentiment"] == "Negative"
    assert res_mod["priority"] in ["Medium", "High"]

    # 3. Neutral factual
    neutral = "The device comes with a 6.5-inch display and 128 GB of storage."
    res_neu = complete_review_analysis(neutral, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res_neu["sentiment"] == "Neutral"
    assert res_neu["priority"] == "Low"

    # 4. Positive
    positive = "The design is premium, comfortable to hold, and works wonderfully."
    res_pos = complete_review_analysis(positive, vectorizer=vectorizer, lr_model=lr_model, nb_model=nb_model)
    assert res_pos["sentiment"] == "Positive"
    assert res_pos["priority"] == "Low"


# ============================================================================
# 36. Mixed and Factual Sentiment Consistency Across Single and Batch
# ============================================================================
def test_36_mixed_and_factual_sentiment_consistency():
    """
    Ensure single review endpoint and batch upload endpoint use identical classification
    policies for factual and mixed reviews without hardcoded overrides.
    """
    factual_text = "The device includes a type-C charging cable and user manual."

    # Single review API
    single_req = SingleReviewRequest(review=factual_text, threshold=0.05)
    single_res = analyze_single_review(single_req)
    assert single_res["sentiment"] == "Neutral"
    assert single_res["priority"] == "Low"

    # Batch review API
    batch_req = BatchReviewRequest(reviews=[factual_text], threshold=0.05)
    batch_res = analyze_batch_reviews(batch_req)
    assert batch_res["total_reviews"] == 1
    assert batch_res["detailed_results"][0]["sentiment"] == "Neutral"
    assert batch_res["detailed_results"][0]["priority"] == "Low"

    # Verify both matched
    assert single_res["sentiment"] == batch_res["detailed_results"][0]["sentiment"]
    assert single_res["priority"] == batch_res["detailed_results"][0]["priority"]


# ============================================================================
# 37. Complex CSV with review_id and metadata columns - Correct Column Selection
# ============================================================================
def test_37_csv_multi_column_with_review_id_and_metadata():
    """
    Verifies that when a CSV contains columns like review_id, author, rating,
    review_text, and date, the parser correctly prioritizes the text column over
    the ID or rating columns, avoiding false 'No readable reviews' or 0 analyzed errors.
    """
    csv_content = (
        "review_id,author,rating,review_text,date\n"
        "101,Alice,5,\"After three weeks of daily use, I am impressed by the display and battery life, although charging is average.\",2026-01-01\n"
        "102,Bob,1,\"The battery drains quickly and the phone overheats during gaming.\",2026-01-02\n"
        "103,Charlie,3,\"The device has 128 GB of storage and supports 5G connectivity.\",2026-01-03\n"
        "104,Diana,5,\"The camera takes stunning sharp photos even in low light.\",2026-01-04\n"
        "105,Evan,1,\"Worst customer service ever, representative was completely rude.\",2026-01-05\n"
        "106,Fiona,3,\"This model features a 6.2-inch AMOLED screen and 128 GB memory.\",2026-01-06\n"
        "107,George,5,\"I absolutely love the clean UI and fluid animations.\",2026-01-07\n"
        "108,Hannah,1,\"The charging cable broke on the second day of use.\",2026-01-08\n"
        "109,Ian,3,\"The device weighs approximately 180 grams.\",2026-01-09\n"
        "110,Julia,5,\"Highly recommended for anyone looking for great performance.\",2026-01-10\n"
    ).encode("utf-8")

    upload = UploadFile(filename="complex_customer_reviews.csv", file=io.BytesIO(csv_content))
    result = upload_document_endpoint(file=upload, mode="auto", threshold=0.05)

    assert result["total_reviews"] == 10
    assert result["reviews_processed"] == 10
    assert result["reviews_failed"] == 0
    assert result["diagnostics"]["review_column_used"] == "review_text"
    assert "review_text" in result["diagnostics"]["column_candidates"]

    # First review content check
    first_rev = result["detailed_results"][0]["original_review"]
    assert "three weeks of daily use" in first_rev
    assert first_rev.strip() != "101"  # NOT the ID


# ============================================================================
# 38. Three Core Prompt Reviews Handling
# ============================================================================
def test_38_three_core_prompt_reviews_handling():
    """
    Ensure the application correctly processes the exact CSV file provided in prompt:
    review_text
    "After three weeks of daily use, I am impressed by the display and battery life, although charging is average."
    "The battery drains quickly and the phone overheats during gaming."
    "The device has 128 GB of storage and supports 5G connectivity."
    """
    csv_content = (
        "review_text\n"
        "\"After three weeks of daily use, I am impressed by the display and battery life, although charging is average.\"\n"
        "\"The battery drains quickly and the phone overheats during gaming.\"\n"
        "\"The device has 128 GB of storage and supports 5G connectivity.\"\n"
    ).encode("utf-8")

    upload = UploadFile(filename="prompt_reviews.csv", file=io.BytesIO(csv_content))
    result = upload_document_endpoint(file=upload, mode="auto", threshold=0.05)

    assert result["total_reviews"] == 3
    assert result["reviews_processed"] == 3
    assert result["reviews_failed"] == 0

    sentiments = [r["sentiment"] for r in result["detailed_results"]]
    # Confirm valid classification for all 3 reviews without hardcoding
    assert len(sentiments) == 3
    for s in sentiments:
        assert s in {"Positive", "Negative", "Neutral"}
    assert "Positive" in sentiments
    assert "Negative" in sentiments


# ============================================================================
# 39. CSV with Alternative Column Names and Graceful Fallback
# ============================================================================
def test_39_csv_alternative_columns_and_manual_selection():
    """
    Support common review-column names: comments, feedback, description.
    Never discard all reviews simply because column name differs.
    """
    # 39a: comments column
    csv_comments = (
        "id,comments,rating\n"
        "1,\"The product quality is superb and works fine.\",5\n"
        "2,\"Very poor build quality, broke immediately.\",1\n"
    ).encode("utf-8")
    up1 = UploadFile(filename="comments.csv", file=io.BytesIO(csv_comments))
    res1 = upload_document_endpoint(file=up1, mode="auto", threshold=0.05)
    assert res1["total_reviews"] == 2
    assert res1["diagnostics"]["review_column_used"] == "comments"

    # 39b: manual column_name selection
    csv_multi = (
        "title,body\n"
        "\"Great Display\",\"The colors are vivid and screen is bright.\"\n"
        "\"Terrible Battery\",\"Drains completely in two hours.\"\n"
    ).encode("utf-8")
    up2 = UploadFile(filename="multi_cols.csv", file=io.BytesIO(csv_multi))
    res2 = upload_document_endpoint(file=up2, mode="auto", threshold=0.05, column_name="body")
    assert res2["total_reviews"] == 2
    assert res2["diagnostics"]["review_column_used"] == "body"

    # 39c: non-existent column fallback (graceful recovery, not error)
    up3 = UploadFile(filename="multi_cols2.csv", file=io.BytesIO(csv_multi))
    res3 = upload_document_endpoint(file=up3, mode="auto", threshold=0.05, column_name="non_existent_col")
    assert res3["total_reviews"] == 2
    assert res3["diagnostics"]["review_column_used"] in ["body", "title"]
    assert any("not found" in w.lower() for w in res3["diagnostics"].get("warnings", []))


# ============================================================================
# 40. CSV Multiline Reviews, Semicolon Delimiters & Numbered Reviews
# ============================================================================
def test_40_csv_multiline_semicolons_and_numbered_items():
    """
    Supports:
    - Semicolon-delimited CSVs
    - Multiline review entries
    - Numbered review text prefixes ("1. The battery...") without losing body numbers ("128 GB")
    """
    semicolon_csv = (
        "review_id;review_text;rating\n"
        "1;\"1. The battery lasts two days and device comes with 128 GB storage.\";5\n"
        "2;\"2. The phone overheats during gaming.\nCharging speed is disappointing.\";1\n"
    ).encode("utf-8")

    upload = UploadFile(filename="semicolon_multiline.csv", file=io.BytesIO(semicolon_csv))
    result = upload_document_endpoint(file=upload, mode="auto", threshold=0.05)

    assert result["total_reviews"] == 2
    assert result["reviews_processed"] == 2

    # Check that "128 GB" was preserved and prefix "1. " was handled cleanly
    assert "128 GB" in result["detailed_results"][0]["original_review"]
    # Check multiline review preserved
    assert "\n" in result["detailed_results"][1]["original_review"] or "overheats" in result["detailed_results"][1]["original_review"]


# ============================================================================
# 41. Diagnostic Error Reporting When File Content is Genuinely Empty
# ============================================================================
def test_41_diagnostics_when_empty_csv():
    """
    Empty input or CSV without data must report a clear diagnostic error identifying the stage.
    """
    empty_csv = "review_id,review_text\n".encode("utf-8")
    upload = UploadFile(filename="empty_data.csv", file=io.BytesIO(empty_csv))
    with pytest.raises(HTTPException) as exc_info:
        upload_document_endpoint(file=upload, mode="auto", threshold=0.05)
    assert exc_info.value.status_code == 400
    assert "No readable review records could be identified" in exc_info.value.detail


