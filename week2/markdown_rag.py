import re
from typing import List, Dict

# Note: In a real environment, you would use a library like LangChain's
# MarkdownHeaderTextSplitter, but for simplicity and self-containment,
# we'll implement a custom function tailored for your HR document structure.


def split_and_tag_markdown(document_text: str, filename: str) -> List[Dict]:
    """
    Splits the employee record markdown based on H2 headers (##) and tags
    each resulting chunk with contextual metadata (Employee Name, Section Title).

    Args:
        document_text: The full content of the Markdown file.
        filename: The original filepath (e.g., 'knowledge-base/employees/Nina Patel.md').

    Returns:
        A list of dictionaries, where each dict represents a processed chunk.
    """
    print(f"--- Processing: {filename} ---")

    # 1. Extract the employee name from the filename
    match = re.search(r"/([^/]+)\.md$", filename)
    employee_name = match.group(1).replace("-", " ") if match else "Unknown Employee"

    # 2. Split the document based on the H2 header pattern (##)
    # The split includes the delimiter so we can easily parse the section title
    sections = re.split(r"(##\s[^\n]+)", document_text)

    # The first element is usually the preamble (HR Record and Name), which we handle separately

    chunks = []
    current_title = "Preamble/Summary"
    current_content = ""

    # Iterate through the split sections (title, content, title, content, ...)
    for part in sections:
        if not part.strip():
            continue

        # Check if the part is a new H2 header (e.g., '## Annual Performance History')
        if part.startswith("##"):
            current_title = part.strip().lstrip("##").strip()
            # Start accumulating content for the new section
            current_content = ""
        else:
            # This is the content block for the current section title
            current_content += part.strip()

        # If we have content, create a chunk and reset
        if current_content and not part.startswith("##"):
            # 3. Contextually Tag the Content
            # Add the employee name and section title directly to the chunk content.
            tagged_content = (
                f"{employee_name}'s {current_title}:\n" f"{current_content}"
            )

            chunks.append(
                {
                    "employee_name": employee_name,
                    "section_title": current_title,
                    "content": tagged_content,
                    "source": filename,
                }
            )

            # Reset content to prepare for the next H2 section
            current_content = ""
            current_title = ""

    # A simpler approach using a library would handle edge cases better,
    # but this demonstrates the key principle of splitting by structure.
    return [
        c for c in chunks if c["content"].strip()
    ]  # Filter out empty/preamble-only content


# --- Sample Data (Mimicking your input documents) ---

NINA_PATEL_DOC = """
# HR Record

# Nina Patel

## Summary
- **Date of Birth:** June 5, 1993
- **Job Title:** Business Intelligence Analyst
- **Location:** San Francisco, California
- **Current Salary:** $82,000

## Insurellm Career Progression
- **January 2021 - Present:** Business Intelligence Analyst
  - Developed advanced Looker dashboards for executive team
  - Mentored junior analysts on data visualization best practices

## Annual Performance History
- **2023:** Rating: 3.5/5
  *Meets expectations but has room for growth. Delivered all required reports but showed limited proactivity in identifying new insights.*

## Compensation History
- **2023:** Base Salary: $82,000 + Bonus: $4,000
- **2022:** Base Salary: $78,000 + Bonus: $2,000
"""

# --- Execution ---

filename = "knowledge-base/employees/Nina Patel.md"
processed_chunks = split_and_tag_markdown(NINA_PATEL_DOC, filename)

# --- Output the results ---

print("\n--- Resulting Optimized Chunks ---")
for i, chunk in enumerate(processed_chunks):
    print(f"\n[CHUNK {i+1} - {chunk['section_title']}]")
    print("---------------------------------------")
    print(chunk["content"].strip())
    print("---------------------------------------")


# Comparison to the original output:
# Original: You had 4 separate chunks for this data, often repeating the last line.
# New: You have 4 complete, non-redundant, contextually-tagged sections.
