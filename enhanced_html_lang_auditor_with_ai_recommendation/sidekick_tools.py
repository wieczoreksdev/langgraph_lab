import asyncio
from locale import normalize
import os
import pandas as pd
import requests
import time
import random
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langcodes import Language, tag_distance, LanguageTagError, standardize_tag
from language_tags import tags
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv(override=True)

#### IANA tool HERE
_IANA_RETRIEVER = None
column_names = ["redirect_info", "current_url", "html_lang_raw", "lang_extracted", "sp_pop", "lang_detected", "match_found", "iana_valid", "status", "recommendation"]

def get_iana_retriever():
    global _IANA_RETRIEVER
    if _IANA_RETRIEVER:
        return _IANA_RETRIEVER

    # Load and chunk registry
    with open("language-subtag-registry.txt", "r", encoding="utf-8") as f:
        text = f.read()
    splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    store = FAISS.from_texts(docs, embeddings)
    _IANA_RETRIEVER = store.as_retriever(search_kwargs={"k": 3})
    return _IANA_RETRIEVER

@tool(description="Use RAG to provide AI recommended language value")
async def query_iana_registry(query: str):
    retriever = get_iana_retriever()
    if not retriever:
        return "Registry not initialized"

    # Use the asynchronous retriever invoke
    docs = await retriever.ainvoke(query)
    #print("\n---\n".join([d.page_content for d in docs]), "HALLLOOOO")
    return "\n---\n".join([d.page_content for d in docs])

###
def get_technical_advice_red(match, iana_valid, html_lang_raw):
    """
    Determines Status and provides a Recommendation.
    """
    recommendation = None
    status = "Manual Check"
    # 1️⃣ Content does not match declared language
    try:
        if html_lang_raw is None:
            status, recommendation ="Critical", "HTML lang attribute is missing."
        elif not match:
            status, recommendation = "Critical", "The detected text language does not match the HTML tag."
        # 2️⃣ Whitespace is always a syntax error
        elif html_lang_raw != html_lang_raw.strip():
            normalized = standardize_tag(html_lang_raw)
            status, recommendation = "Fix", f"Syntax error (whitespace). Change '{html_lang_raw}' to '{normalized}'"
        # 3️⃣ IANA / BCP 47 invalid
        elif not iana_valid:
            normalized = standardize_tag(html_lang_raw)
            status, recommendation = "Fix", f"Invalid BCP 47 syntax. Change '{html_lang_raw}' to '{normalized}'"
        # 4️⃣ Only consider canonicalization if IANA-valid
        elif iana_valid:
            normalized = standardize_tag(html_lang_raw)
            if html_lang_raw != normalized:
                status, recommendation = "Keep", f"Valid, but standard suggests '{normalized}'"
            else:
                # 5️⃣ Fully correct
                 status, recommendation = "Keep", "Perfect"
        
        
    except Exception as e:
        status = "Error"
        recommendation = f"Exception occurred: {type(e).__name__}"
    else:
        # This runs if NO EXCEPTION was raised
        # You can use this for logging successful audits
        print(f"Audit successful for: {html_lang_raw}")
    finally:
        # This ALWAYS runs. Great for final cleanup or formatting.
        # Ensure we always return a clean tuple
        return status, recommendation

# COPY
# def get_technical_advice_red(match, iana_valid, html_lang_raw):
#     """
#     Determines Status and provides a Recommendation.
#     """
#     # 1️⃣ Content does not match declared language
#     try:
#         if not match:
#             return "Critical", "The detected text language does not match the HTML tag."
#         # 2️⃣ Whitespace is always a syntax error
#         if html_lang_raw != html_lang_raw.strip():
#             normalized = standardize_tag(html_lang_raw)
#             return (
#                 "Fix",
#                 f"Syntax error (whitespace). Change '{html_lang_raw}' to '{normalized}'"
#             )
#         # 3️⃣ IANA / BCP 47 invalid
#         if not iana_valid:
#             normalized = standardize_tag(html_lang_raw)
#             return (
#                 "Fix",
#                 f"Invalid BCP 47 syntax. Change '{html_lang_raw}' to '{normalized}'"
#             )
#         # 4️⃣ Only consider canonicalization if IANA-valid
#         if iana_valid:
#             normalized = standardize_tag(html_lang_raw)
#             print(f'NORMALIZED {normalized}')
#             if html_lang_raw != normalized:
#                 return (
#                     "Keep",
#                     f"Valid, but standard suggests '{normalized}'"
#                 )
#             else:
#                 # 5️⃣ Fully correct
#                 print(f'FULLY CORRECT{normalized}')
#                 return "Keep", "Perfect"
#         # 6️⃣ Fallback (catch-all)
#         return "Manual Check", "Check"
#     except Exception:
#         return  "Error has occured", "Check data"

def get_technical_advice_green(match, iana_valid, html_lang_raw):
    """
    Determines Status and provides a Recommendation.
    """
    if not match:
        return "Critical", "The detected text language does not match the HTML tag."
    
    if iana_valid:
        # Check if it's valid but could be better (e.g., pl-pl -> pl-PL)
        try:
            normalized = standardize_tag(html_lang_raw)
            if html_lang_raw != normalized:
                return "Keep", f"Valid, but standard suggests '{normalized}'"
            return "Keep", "Perfect"
        except:
            return "Check", "Issue"
    
    #f not IANA valid, suggest a fix
    try:
        normalized = standardize_tag(html_lang_raw)
        return "Fix", f"Syntax error. Change '{html_lang_raw}' to '{normalized}'"
    except:
        return "Fix", "Invalid BCP 47 syntax. Check for spaces or illegal characters."

def get_url_lang_data_langcodes(url: str):
    # Default values
    current_url = url
    html_lang_raw, lang_extracted, lang_detected, sp_pop, redirect_info, lang_obj = None, None, None, None, None, None,
    match_found, iana_valid = False, False
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
                    print(f'INSIDE try lang_obj = Language.get(html_lang_raw, normalize=False):{lang_obj}, {lang_obj.language}, {iana_valid}')
                    #print(f"en-DZ is valid?: {Language.get('en-DZ').is_valid()}")
                    
                    lang_extracted = lang_obj.language 
                except (LanguageTagError, Exception):
                    iana_valid, lang_extracted = False, "invalid"

            # 3. Text Extraction
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)

            if len(text) < 50:
                lang_detected, match_found = "insufficient_text", False
            else:
                try:
                    lang_detected = detect(text).lower()
                    # 4. Strict Distance Check (No stripping)
                    try:
                        dist = tag_distance(html_lang_raw, lang_detected)
                        match_found = (dist < 10)
                    except:
                        match_found = (lang_extracted == lang_detected)
                except LangDetectException:
                    lang_detected, match_found = "detection_error", False
            # 5. Get Status and Recommendation
            status, recommendation = get_technical_advice_red(match_found, iana_valid, html_lang_raw)
            sp_pop = Language.get(html_lang_raw, normalize=True).speaking_population() 
            return (redirect_info, current_url, html_lang_raw, lang_extracted, sp_pop, lang_detected, 
                    match_found, iana_valid, status, recommendation)
    except Exception as e:
        return (str(type(e).__name__), url, "Error html_lang_raw", "Error lang_extracted","Error sp_pop",  "Error lang_detected", False, False, "Error Status", "Error recommendation")

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
    local_column_names = [*column_names]
    
    results_df = pd.DataFrame(all_results, columns=local_column_names)
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

def get_url_lang_data_language_tags(url: str):
    # Initialize defaults to ensure return consistency
    current_url, html_lang, lang_extracted, lang_detected = url, "n/a", "n/a", "n/a"
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
                lang_extracted = html_lang.split("-")[0].lower()
            
            # 2. Extract and Detect Text Language
            for script in soup(["script", "style", "noscript"]):
                script.decompose()
            text = soup.get_text(separator=" ", strip=True)

            if len(text) < 50:
                lang_detected = "insufficient_text"
            else:
                try:
                    lang_detected = detect(text).lower()
                    match_found = (lang_extracted == lang_detected)
                except LangDetectException:
                    lang_detected = "detection_error"

            status = getStatus(match_found, iana_valid)
            return (current_url, html_lang, lang_extracted, lang_detected, match_found, iana_valid, status, redirect_info)

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
    local_column_names = [*column_names]
    csv_df[local_column_names] = results
    
    # Save the result
    output_path = os.path.join("./.gradio/output", "audit_result.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    csv_df.to_csv(output_path, index=False)
    
    print(f"Audit complete. Results saved to: {output_path}")
    return output_path


# https://docs.langchain.com/oss/python/integrations/tools/google_serper
serper = GoogleSerperAPIWrapper()
@tool(description="Search the web using Google Serper API")
def search_web(query: str):
    return serper.run(query)

def all_tools():
    #file_tools = file_management_tools()
    #playwright_tools, _ = await setup_playwright()
    return [page_language_tags_auditor, page_langcodes_auditor, search_web]

if __name__ == "__main__":
    # tools = asyncio.run(all_tools())
    # print(f"Loaded {tools} tools.")
    print(get_url_lang_data_langcodes("https://www.pmi.com/markets/poland/pl/about-us/overview/"))

    # def get_technical_advice(match, iana_valid, html_lang_raw, lang_obj):
#     """
#     Determines Status and provides a Recommendation.
#     """

#     # 1️⃣ Content does not match declared language
#     if not match:
#         return "Critical", "The detected text language does not match the HTML tag."

#     # 2️⃣ Whitespace is always a syntax error
#     if html_lang_raw != html_lang_raw.strip():
#         normalized = standardize_tag(html_lang_raw)
#         return (
#             "Fix",
#             f"Syntax error (whitespace). Change '{html_lang_raw}' to '{normalized}'"
#         )

#     # 3️⃣ IANA / BCP 47 invalid
#     if not iana_valid:
#         normalized = standardize_tag(html_lang_raw)
#         return (
#             "Fix",
#             f"Invalid BCP 47 syntax. Change '{html_lang_raw}' to '{normalized}'"
#         )

#     # 4️⃣ Valid but not canonical (case / order)
#     normalized = standardize_tag(html_lang_raw)
#     if html_lang_raw != normalized:
#         return (
#             "Keep",
#             f"Valid, but standard suggests '{normalized}'"
#         )

#     # 5️⃣ Fully correct
#     return "Keep", "Perfect"

# def get_technical_advice(match, iana_valid, html_lang_raw, lang_obj):
#     """
#     Determines Status and provides a Recommendation.
#     """
#     if not match:
#         return "Critical", "The detected text language does not match the HTML tag."
    
#     if iana_valid:
#         # Check if it's valid but could be better (e.g., pl-pl -> pl-PL)
#         try:
#             normalized = standardize_tag(html_lang_raw)
#             if html_lang_raw != normalized:
#                 return "Keep", f"Valid, but standard suggests '{normalized}'"
#             return "Keep", "Almost Perfect"
#         except:
#             return "Keep", "Perfect"
    
#     # If not IANA valid, suggest a fix
#     try:
#         normalized = standardize_tag(html_lang_raw)
#         return "Fix", f"Syntax error. Change '{html_lang_raw}' to '{normalized}'"
#     except:
#         return "Fix", "Invalid BCP 47 syntax. Check for spaces or illegal characters."
