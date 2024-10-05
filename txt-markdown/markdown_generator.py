import json
import sys
import os
from bs4 import BeautifulSoup
import html2text
import streamlit as st


def extract_qa_markdown(json_data):
    """
    Extracts questions, answers, and URLs from JSON data and formats them in Markdown.

    Args:
        json_data (dict): The JSON data containing articles.

    Returns:
        str: The extracted Q&A with URLs in Markdown format.
    """
    markdown_output = ""
    h = html2text.HTML2Text()
    h.ignore_images = False  # Set to False to include images
    h.ignore_links = False
    h.bypass_tables = False
    h.ignore_emphasis = False
    h.ignore_tables = True
    h.body_width = 0  # Prevents line wrapping

    for article in json_data.get('data', []):
        title = article.get('title', 'No Title')
        url = article.get('url', '')
        markdown_output += f"## {title}\n\n"

        if url:
            # Add the URL as a clickable link
            markdown_output += f"[View Article]({url})\n\n"

        body = article.get('body', '')
        soup = BeautifulSoup(body, 'html.parser')

        # Find all question headers (assuming they are <h1> to <h6> tags)
        questions = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])

        for q in questions:
            question_text = q.get_text(strip=True)
            # Convert question header to Markdown (e.g., ### Q1. ...)
            markdown_output += f"### {question_text}\n\n"

            # Initialize a list to hold answer elements
            answer_markdown = ""

            # Start with the next sibling of the question
            sibling = q.find_next_sibling()

            # Traverse siblings until the next question or end of content
            while sibling and sibling.name not in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                if sibling.name == 'ul':
                    # Convert <ul> to Markdown list
                    list_markdown = h.handle(str(sibling))
                    answer_markdown += list_markdown + "\n"
                elif sibling.name == 'p':
                    # Convert <p> to Markdown paragraph
                    para_markdown = h.handle(str(sibling))
                    answer_markdown += para_markdown + "\n"
                elif sibling.name == 'img':
                    # Convert <img> to Markdown image
                    img_url = sibling.get('src', '')
                    if img_url:
                        answer_markdown += f"![Image]({img_url})\n\n"
                # Move to the next sibling
                sibling = sibling.find_next_sibling()

            markdown_output += answer_markdown + "\n"

        markdown_output += "\n---\n\n"  # Separator between articles

    return markdown_output

def main():
    st.title("Extract Q&A from JSON to Markdown")
    """
    Main function to handle file input and output.
    """

    txt_file = st.file_uploader("", type=["txt"])

    if txt_file is not None:
        file_contents = txt_file.read().decode("utf-8")

        try:

            # Parse the JSON data
            json_data = json.loads(file_contents)

            # Extract Q&A in Markdown
            markdown = extract_qa_markdown(json_data)

            st.download_button(
                label="Download Markdown File",
                data=markdown,
                file_name='output.md',
                mime='text/markdown'
            )

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
