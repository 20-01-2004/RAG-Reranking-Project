from pdf_loader import load_pdf

pdf_path = "data/documents/sample.pdf"

pages = load_pdf(pdf_path)

print(f"Number of pages extracted: {len(pages)}")

for page in pages:
    print("\n--------------------")
    print("Source:", page["source"])
    print("Page:", page["page"])
    print("Text:")
    print(page["text"][:500])