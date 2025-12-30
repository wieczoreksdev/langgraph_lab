import asyncio
import os
import pandas as pd
import requests
import time
import random
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from playwright.async_api import async_playwright
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_experimental.utilities import PythonREPL
from tempfile import TemporaryDirectory
from langchain_community.agent_toolkits import FileManagementToolkit
from langcodes import Language, tag_distance, LanguageTagError
from language_tags import tags
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(override=True)

#### IANA tool HERE
# def get_iana_retriever():
#     registry_path = "language-subtag-registry.txt"
#     if not os.path.exists(registry_path):
#         return None
    
#     with open(registry_path, "r", encoding="utf-8") as f:
#         content = f.read()
    
#     # The IANA registry uses %% as a record separator
#     docs = content.split("%%")
    
#     vectorstore = FAISS.from_texts(
#         texts=docs,
#         embedding=OpenAIEmbeddings()
#     )
#     return vectorstore.as_retriever(search_kwargs={"k": 2})

# @tool
# def query_iana_registry(query: str):
#     """Searches the official IANA language subtag registry for specific tags, 
#     descriptions, or syntax rules. Use this to find the correct 'ai_recommended' tag."""
#     retriever = get_iana_retriever()
#     if not retriever:
#         return "Registry file not found."
#     results = retriever.invoke(query)
#     return "\n---\n".join([r.page_content for r in results])
####

def get_technical_advice(match, iana_valid, html_lang_raw, lang_obj):
    """
    Determines Status and provides a Recommendation.
    """

    # 1️⃣ Content does not match declared language
    if not match:
        return "Critical", "The detected text language does not match the HTML tag."

    # 2️⃣ Whitespace is always a syntax error
    if html_lang_raw != html_lang_raw.strip():
        normalized = str(lang_obj.normalize())
        return (
            "Fix",
            f"Syntax error (whitespace). Change '{html_lang_raw}' to '{normalized}'"
        )

    # 3️⃣ IANA / BCP 47 invalid
    if not iana_valid:
        normalized = str(lang_obj.normalize())
        return (
            "Fix",
            f"Invalid BCP 47 syntax. Change '{html_lang_raw}' to '{normalized}'"
        )

    # 4️⃣ Only consider canonicalization if IANA-valid
    if iana_valid:
        normalized = str(lang_obj.normalize())
        if html_lang_raw != normalized:
            return (
                "Keep",
                f"Valid, but standard suggests '{normalized}'"
            )
        else:
            # 5️⃣ Fully correct
            return "Keep", "Perfect"

    # 6️⃣ Fallback (catch-all)
    return "Manual Check", "Check"
    
# def get_technical_advice(match, iana_valid, html_lang_raw, lang_obj):
#     """
#     Determines Status and provides a Recommendation.
#     """
#     if not match:
#         return "Critical", "The detected text language does not match the HTML tag."
    
#     if iana_valid:
#         # Check if it's valid but could be better (e.g., pl-pl -> pl-PL)
#         try:
#             normalized = str(lang_obj.normalize())
#             if html_lang_raw != normalized:
#                 return "Keep", f"Valid, but standard suggests '{normalized}'"
#             return "Keep", "Perfect"
#         except:
#             return "Keep", "Perfect"
    
#     # If not IANA valid, suggest a fix
#     try:
#         normalized = str(lang_obj.normalize())
#         return "Fix", f"Syntax error. Change '{html_lang_raw}' to '{normalized}'"
#     except:
#         return "Fix", "Invalid BCP 47 syntax. Check for spaces or illegal characters."

def get_url_lang_data_langcodes(url: str):
    # Default values
    current_url, html_lang_raw, norm_lang, det_lang = url, "Not found", "n/a", "n/a"
    match_found, iana_valid, lang_obj = False, False, None
    redirect_info = "n/a"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        current_url = response.url
        redirect_info = f"{response.status_code}{'_redirected' if len(response.history) > 0 else ''}"

        if response.text:
            soup = BeautifulSoup(response.text, "html.parser")
            html_tag = soup.find("html")
            
            # 1. RAW Tag Capture
            if html_tag and html_tag.has_attr("lang"):
                html_lang_raw = html_tag.get("lang", "") 
                
                # 2. Strict Validation
                try:
                    lang_obj = Language.get(html_lang_raw, normalize=False)
                    iana_valid = lang_obj.is_valid()

                    print(f'pl-EN is valid?: {Language.get('pl-EN').is_valid()}')
                    
                    norm_lang = lang_obj.language 
                except (LanguageTagError, Exception):
                    iana_valid, norm_lang = False, "invalid"

            # 3. Text Extraction
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)

            if len(text) < 50:
                det_lang, match_found = "insufficient_text", False
            else:
                try:
                    det_lang = detect(text).lower()
                    # 4. Strict Distance Check (No stripping)
                    try:
                        dist = tag_distance(html_lang_raw, det_lang)
                        match_found = (dist < 10)
                    except:
                        match_found = (norm_lang == det_lang)
                except LangDetectException:
                    det_lang, match_found = "detection_error", False

            # 5. Get Status and Recommendation
            status, recommendation = get_technical_advice(match_found, iana_valid, html_lang_raw, lang_obj)
            
            return (current_url, html_lang_raw, norm_lang, det_lang, 
                    match_found, iana_valid, status, recommendation, redirect_info)

    except Exception as e:
        return (url, "Error", "error", "error", False, False, "Error Status", "Connection Failed", str(type(e).__name__))

@tool(description="Audit page language tags with a random delay between requests to bypass site monitoring.")
def page_langcodes_auditor(path: str, output_folder_path:str):
    if not os.path.exists(path):
        return f"[File not found: {path}]"
    
    print(f"Starting Professional Throttled Audit on: {path}")
    csv_df = pd.read_csv(path)
    
    if 'url' not in csv_df.columns:
        return "Error: CSV must have a 'url' column."

    all_results = []
    total = len(csv_df)
    
    # Professional Loop Implementation
    for index, row in csv_df.iterrows():
        url = row['url']
        print(f"Progress: [{index + 1}/{total}] Processing: {url}")
        
        # 1. Hit the page
        res = get_url_lang_data_langcodes(url)
        all_results.append(res)
        
        # 2. Random Delay to mimic human browsing (1.0 to 2.5 seconds)
        if index < total - 1: # Don't wait after the very last one
            delay = random.uniform(1.0, 2.5)
            time.sleep(delay)
    
    # Assemble the final data
    column_names = [
        "current_url", "html_lang_raw", "normalized_lang", 
        "lang_detected", "is_match", "is_iana_valid", 
        "status", "recommendation", "redirect_info"
    ]
    
    results_df = pd.DataFrame(all_results, columns=column_names)
    final_df = pd.concat([csv_df.reset_index(drop=True), results_df], axis=1)
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
    # Save Output
    output_path = os.path.join(output_folder_path, f'{timestamp}_result.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False)
    
    print(f"Audit Complete. Results: {output_path}")
    return output_path


def getStatus(match, iana_valid):
    """
    Critical: The text on the page doesn't match the declared tag.
    Fix: The language matches, but the tag is technically invalid (spaces, wrong format).
    Keep: Everything is perfect.
    """
    if not match:
        return "Critical"
    return "Keep" if iana_valid else "Fix"
\
def get_url_lang_data_language_tags(url: str):
    # Initialize defaults to ensure return consistency
    current_url, html_lang, norm_lang, det_lang = url, "n/a", "n/a", "n/a"
    match_found, iana_valid = False, False
    redirect_info = "unknown"

    try:
        response = requests.get(url, timeout=10)
        current_url = response.url
        status_code = response.status_code
        redirect_occurred = len(response.history) > 0
        redirect_info = f"{status_code}{'_redirected' if redirect_occurred else ''}"

        if response.text:
            soup = BeautifulSoup(response.text, "html.parser")
            html_tag = soup.find("html")
            
            # 1. Extract and Validate HTML Lang
            if html_tag and html_tag.has_attr("lang"):
                html_lang = html_tag.get("lang", "").strip()
                iana_valid = tags.check(html_lang)
                norm_lang = html_lang.split("-")[0].lower()
            
            # 2. Extract and Detect Text Language
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            text = soup.get_text(separator=" ", strip=True)

            if len(text) < 50:
                det_lang = "insufficient_text"
            else:
                try:
                    det_lang = detect(text).lower()
                    match_found = (norm_lang == det_lang)
                except LangDetectException:
                    det_lang = "detection_error"

            status = getStatus(match_found, iana_valid)
            return (current_url, html_lang, norm_lang, det_lang, match_found, iana_valid, status, redirect_info)

        return (current_url, "no_content", "n/a", "n/a", False, False, "Error Status", redirect_info)

    except Exception as e:
        error_name = str(type(e).__name__).lower()
        return (url, "error", "error", "error", False, False, "Error Status", error_name)


@tool(description="Audit page language tags and compare with detected text language with language_tags")
def page_language_tags_auditor(path: str):
    if not os.path.exists(path):
        return f"[File not found: {path}]"
    
    print(f"Starting 'Zero Mercy' Audit on: {path}")
    csv_df = pd.read_csv(path)
    
    # Process all URLs in the CSV
    results = csv_df['url'].apply(lambda x: pd.Series(get_url_lang_data_language_tags(x)))
    
    # Define column names for the output
    column_names = [
        "current_url", "html_lang_raw", "normalized_lang", 
        "lang_detected", "is_match", "is_iana_valid", 
        "status", "redirect_info"
    ]
    csv_df[column_names] = results
    
    # Save the result
    output_path = os.path.join("./.gradio/output", "audit_result.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csv_df.to_csv(output_path, index=False)
    
    print(f"Audit complete. Results saved to: {output_path}")
    return output_path

# https://docs.langchain.com/oss/python/integrations/tools/python
python_repl = PythonREPL()
@tool(description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.")
def run_python(code: str):
    return python_repl.run(code)
# python_repl_tool = Tool(
#     name="python_repl",
#     description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
#     func=python_repl.run,
# )

pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"
@tool(description="Send a push notification to the user")
def push_notification(text: str):
    requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})
    return "success"

# https://docs.langchain.com/oss/python/integrations/tools/wikipedia
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
@tool(description="Search Wikipedia for relevant information")
def wiki(query: str):
    return wikipedia.run(query)

# https://docs.langchain.com/oss/python/integrations/tools/google_serper
serper = GoogleSerperAPIWrapper()
@tool(description="Search the web using Google Serper API")
def search_web(query: str):
    return serper.run(query)
# search_web = Tool(
#     name="search_web",
#     description="Search the web using Google Serper API",
#     func=serper.run
# )

# https://docs.langchain.com/oss/python/integrations/tools/playwright

async def setup_playwright():
    # Explicit async context to avoid nested event loop
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools = toolkit.get_tools()
        return tools, browser 


# https://docs.langchain.com/oss/python/integrations/tools/filesystem
working_directory = TemporaryDirectory()


def file_management_tools():
    print(working_directory, "WORKING DIRECTORY")
    toolkit = FileManagementToolkit(
        root_dir=working_directory.name
    )  # If you don't provide a root_dir, operations will default to the current working directory
    return  toolkit.get_tools()

@tool(description="Return the CSV header and first data row as a string snippet for context.")
def read_csv_snippet(file_path: str) -> str:
    if not os.path.exists(file_path):
        return "[File not found]"

    try:
        df = pd.read_csv(file_path)
        if df.empty:
            return "[CSV is empty]"
        # Select only header + first data row
        snippet_df = df.head(1)  # first row (after header)
        # Convert to CSV string with header
        snippet = snippet_df.to_csv(index=False)
        return snippet
    except Exception as e:
        return f"[Unable to read file: {e}]"


async def all_tools():
    file_tools = file_management_tools()
    playwright_tools, _ = await setup_playwright()
    return file_tools + playwright_tools + [search_web, wiki, push_notification, run_python, read_csv_snippet, page_language_tags_auditor, page_langcodes_auditor]

if __name__ == "__main__":
    # tools = asyncio.run(all_tools())
    # print(f"Loaded {tools} tools.")
    print(get_url_lang_data_langcodes("https://www.pmi.com/markets/poland/pl/about-us/overview/"))
    

