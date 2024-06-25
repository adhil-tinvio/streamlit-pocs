import streamlit as st
import pandas as pd
import requests
import concurrent.futures
from collections import defaultdict
from fuzzywuzzy import fuzz
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = st.secrets["API_KEY"]
ACTIVE_ONLY_ACCOUNTS = ['Input GST Receivable',
                        'Input GST Payable',
                        'Accounts Payable',
                        'FX Realized Currency (Gains)/Loss',
                        'FX Unrealized Currency (Gains)/Loss',
                        'FX Bank Revaluation (Gains)/Loss',
                        'FX Rounding (Gains)/Loss',
                        'FX Realized Currency (Gains)/Losses',
                        'FX Unrealized Currency (Gains)/Losses',
                        'FX Bank Revaluation (Gains)/Losses',
                        'FX Rounding (Gains)/Losses',
                        'Business Bank Account',
                        'Accounts Receivable',
                        'Retained Earnings',
                        'Input VAT Receivable',
                        'Output VAT Payable']

COLUMNS_WITH_CURRENCY = ['Account Type*',
                         'Currency*',
                         'Name*',
                         'Code',
                         'Description',
                         'Lock Date',
                         'Status',
                         'Controlled Account (do not edit)',
                         'Unique ID (do not edit)']

COLUMNS_WITHOUT_CURRENCY = ['Account Type*',
                            'Name*',
                            'Code',
                            'Description',
                            'Lock Date',
                            'Status',
                            'Controlled Account (do not edit)',
                            'Unique ID (do not edit)']

jaz_account_type_mappings = {
    "Current Liability": "Liability - Current Liability",
    "Bank Accounts": "Asset - Bank Accounts",
    "Operating Expense": "Expense - Operating Expense",
    "Current Asset": "Asset - Current Asset",
    "Direct Costs": "Expense - Direct Costs",
    "Shareholders Equity": "Equity - Shareholders Equity",
    "Fixed Asset": "Asset - Fixed Asset",
    "Non-current Asset": "Asset - Non-current Asset",
    "Non-current Liability": "Liability - Non-current Liability",
    "Other Revenue": "Revenue - Other Revenue",
    "Operating Revenue": "Revenue - Operating Revenue",
    "Inventory": "Asset - Inventory",
    "Cash": "Asset - Cash"
}

external_account_type_mappings = {
    "Accounts Payable": "Liability - Current Liability",
    "Accounts Receivable": "Asset - Current Asset",
    "Bank": "Asset - Bank Accounts",
    "Bank Revaluations": "Expense - Operating Expense",
    "Current Asset": "Asset - Current Asset",
    "Current Liability": "Liability - Current Liability",
    "Direct Costs": "Expense - Direct Costs",
    "Equity": "Equity - Shareholders Equity",
    "Expense": "Expense - Operating Expense",
    "Fixed Asset": "Asset - Fixed Asset",
    "Historical Adjustment": "Liability - Current Liability",
    "Non-Current Asset": "Asset - Non-current Asset",
    "Non-current Liability": "Liability - Non-current Liability",
    "Other Income": "Revenue - Other Revenue",
    "Realized Currency Gains": "Expense - Operating Expense",
    "Retained Earnings": "Equity - Shareholders Equity",
    "Revenue": "Revenue - Operating Revenue",
    "Rounding": "Liability - Current Liability",
    "Sales Tax": "Liability - Current Liability",
    "Tracking": "Liability - Current Liability",
    "Unpaid Expense Claims": "Liability - Current Liability",
    "Unrealized Currency Gains": "Expense - Operating Expense",
    "Wages Payable": "Liability - Current Liability",
    "Inventory": "Asset - Inventory",
    "Cash": "Asset - Cash"
}


def validate_sga_match_response(value):
    if value in ['No Suitable Match', 'Error in classification', '']:
        return False
    if pd.isna(value):
        return False
    return True


# Function to return the right column value
def get_account_type_mapping(external_account_type):
    for key, value in external_account_type_mappings.items():
        if fuzz.ratio(key, external_account_type) > 95:
            return value

    for key, value in jaz_account_type_mappings.items():
        if fuzz.ratio(key, external_account_type) > 95:
            return value

    return f'Not Mapped: {external_account_type}'


account_type_prompt = """
You are a skilled accountant. Determine the closest matching Accepted Account Type for each Account Type or Report Code combination given as input below. 
If there is no matching type, return an empty value

The accepted account types are:
1. Asset - Bank Accounts 
2. Asset - Cash
3. Asset - Current Asset
4. Asset - Fixed Asset
5. Asset - Inventory
6. Asset - Non-current Asset
7. Equity - Shareholders Equity
8. Expense - Direct Costs
9. Expense - Operating Expense
10. Expense - Other Expense
11. Liability - Current Liability
12. Liability - Non-current Liability
13. Revenue - Operating Revenue
14. Revenue - Other Revenue

Only if the account type was not sufficient to determine the correct accepted account type, then use the report code to 
determine the accepted account type. Here is a help abbreviation glossary that maps to the accepted account types:
EXP = Expense - Operating Expense
EQU = Equity - Shareholders Equity

Criteria:
1) Return only Accepted Account Type from the list above as response. If there is no close match,return ''.
2) Please do not give them index numbers at all.
3) Make sure the return list length is exactly the same as the input size length
 (VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
4) Please do not have empty lines in your return. The results should all be in the next line 
(VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
"""


def classify_account_types(account_types, report_codes, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(account_types)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(account_types))
        account_type_batch = account_types[start_index:end_index]
        report_code_batch = report_codes[start_index:end_index]
        messages = [{
            'role': 'system',
            'content': account_type_prompt
        }]
        for i in range(len(account_type_batch)):
            messages.append({
                'role': 'user',
                'content': f"Account Type: {account_type_batch[i]}  Report Code: {report_code_batch}"
            })

        data = {
            "model": "gpt-4-turbo",
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 1000
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            content = content.strip("```").strip()
            batch_results = [line.strip() for line in content.split('\n')]

            if len(batch_results) != len(account_type_batch):
                print("batch: ", account_type_batch)
                print("batch results: ", batch_results)
                raise ValueError(f"Expected {len(account_type_batch)} results, but got {len(batch_results)}")
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            batch_results = ["Error in classification"] * len(account_type_batch)

        results[start_index:end_index] = batch_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices = range(0, len(account_types), batch_size)
        executor.map(process_batch, indices)

    return results


def sga_prompt_generator(chart_of_accounts):
    sga_prompt = """
    You are a skilled financial analyst.
    Match the given account names with the most suitable account from the Chart of Accounts given to you.
    Below is the list of available Chart of Accounts in the format:  {'Account Name','Account Type'}
    The input will also be similar format {'Account Name','Account Type'}
    """ + "\n- " + "\n- ".join(chart_of_accounts) + "\n"

    sga_prompt += """
    1) Return only Account Name as response,If there is no close match,Return ONLY 'No Suitable Match' (VERY IMPORTANT).
    2) Make sure the return list length is exactly the same as the input size length (VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
    3) The matches must be 1:1, meaning each Account Name from the list must be paired uniquely with 
    one Account from the Chart of Accounts given and vice versa (VERY IMPORTANT).
    4) Please do not give them index numbers at all.
    5) Please do not have empty lines in your return. The results should all be in the next line IMPORTANT
    6) the Account Type when matched should not be conflicting
    """

    return sga_prompt


def recommend_sga_match(jaz_account_details, ext_coa_account_details, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(ext_coa_account_details)
    sga_prompt = sga_prompt_generator(jaz_account_details)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(ext_coa_account_details))
        ext_ac = ext_coa_account_details[start_index:end_index]
        messages = [{'role': 'system', 'content': sga_prompt}]
        for t in range(len(ext_ac)):
            messages.append({'role': 'user', 'content': ext_ac[t]})

        data = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 1000
        }

        try:
            response = requests.post('https://api.openai.com/v1/chat/completions', headers=headers, json=data)
            response.raise_for_status()
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            content = content.strip("```").strip()
            batch_results = [line.strip() for line in content.split('\n') if line.strip()]

            if len(batch_results) != len(ext_ac):
                print("batch: ", ext_ac)
                print("batch results: ", batch_results)
                raise ValueError(f"Expected {len(ext_ac)} results, but got {len(batch_results)}")
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            batch_results = ["Error in classification"] * len(ext_ac)

        results[start_index:end_index] = batch_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices = range(0, len(ext_coa_account_details), batch_size)
        executor.map(process_batch, indices)

    return results


def match_coa_using_gpt(external_coa_df, jaz_coa_df, jaz_coa_map, mapped_coa_names, code_flag, desc_flag):
    unmapped_external_coa = external_coa_df[~(external_coa_df['Name'].isin(mapped_coa_names))]
    jaz_controlled_account_names = []
    jaz_account_types = []
    for i in range(len(jaz_coa_df)):
        controlled_account_name = jaz_coa_df.iloc[i]['Controlled Account (do not edit)']
        account_name = jaz_coa_df.iloc[i]['Name*']
        account_type = jaz_coa_df.iloc[i]['Account Type*']
        if (controlled_account_name == "" or controlled_account_name is None
                or jaz_coa_map[account_name]['Match'] or pd.isna(controlled_account_name)):
            continue
        else:
            jaz_controlled_account_names.append(controlled_account_name)
            jaz_account_types.append(account_type)
    ext_coa_account_names = unmapped_external_coa['Name'].tolist()
    ext_coa_account_types = unmapped_external_coa['Type'].tolist()
    jaz_account_details = [f"{{'Account Name': {controlled_account_name} , 'Account Type': {account_type}}}"
                           for controlled_account_name, account_type in
                           zip(jaz_controlled_account_names, jaz_account_types)]
    ext_coa_account_details = [f"{{'Account Name': {coa_account_name} , 'Account Type': {coa_account_type}}}"
                               for coa_account_name, coa_account_type in
                               zip(ext_coa_account_names, ext_coa_account_types)]
    sga_matches = recommend_sga_match(jaz_account_details, ext_coa_account_details, 10)
    st.write("sga_op", sga_matches)
    st.write("jaz_account_details", jaz_account_details)
    st.write("ext_coa_account_details", ext_coa_account_details)
    st.write("coa_account_names", ext_coa_account_names)
    st.write("coa_account_typees", ext_coa_account_types)
    if len(sga_matches) != len(ext_coa_account_names):
        return jaz_coa_map, mapped_coa_names
    sga_conflict_map = defaultdict(int)
    for i in range(len(sga_matches)):
        value = sga_matches[i]
        sga_conflict_map[value] += 1
    for i in range(len(sga_matches)):
        if validate_sga_match_response(sga_matches[i]) and sga_conflict_map[sga_matches[i]] == 1:
            jaz_coa_controlled_account_name = sga_matches[i]
            ext_coa_name = ext_coa_account_names[i]
            filtered_df = external_coa_df[external_coa_df['Name'] == ext_coa_name]
            if len(filtered_df) > 0:
                filtered_row = filtered_df.iloc[0]
                for elem, value in jaz_coa_map.items():
                    if (value['Controlled Account (do not edit)'] is not None and
                            value['Controlled Account (do not edit)'] == jaz_coa_controlled_account_name):
                        jaz_coa_map[elem]['Name*'] = filtered_row['Name']
                        jaz_coa_map[elem]['Account Type*'] = filtered_row['Type']
                        if code_flag:
                            jaz_coa_map[elem]['Code'] = filtered_row['Code']
                        if desc_flag:
                            jaz_coa_map[elem]['Description'] = filtered_row['Description']
                        jaz_coa_map[elem]['Match'] = True
                        jaz_coa_map[elem]['Status'] = 'ACTIVE'
                        jaz_coa_map[elem]['Match Type'] = 'GPT'
                        mapped_coa_names.add(ext_coa_name)
    return jaz_coa_map, mapped_coa_names


def convert_df_to_csv(df):
    csv_df = df.to_csv(index=False, encoding='utf-8')
    return csv_df


def contains_substring(substring, text):
    """
    This function checks if the given text contains the specified substring.

    Args:
    text (str): The text to search within.
    substring (str): The substring to search for.

    Returns:
    bool: True if the substring is found in the text, False otherwise.
    """
    # Escape the substring to handle special regex characters
    pattern = re.escape(substring)

    # Search for the pattern in the text
    match = re.search(pattern, text, re.IGNORECASE)  # Adding re.IGNORECASE for case insensitive match

    # Return True if a match is found, otherwise False
    return bool(match)


def calculate_cosine_similarity(text1, text2):
    # Example documents
    documents = [text1, text2]

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform documents to TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Compute cosine similarity between the two TF-IDF vectors
    similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]*100

    return similarity


def update_external_coa_column_names(external_coa_df):
    external_coa_columns = external_coa_df.columns
    name_column = None
    type_column = None
    code_column = None
    description_column = None
    for i in range(len(external_coa_columns)):
        if external_coa_columns[i] == 'jaz_sga_name':
            continue
        if calculate_cosine_similarity("Account Name", external_coa_columns[i]) >= 50:
            name_column = external_coa_columns[i]
        elif calculate_cosine_similarity("Account Type", external_coa_columns[i]) >= 50:
            type_column = external_coa_columns[i]
        elif calculate_cosine_similarity("Code", external_coa_columns[i]) >= 50:
            code_column = external_coa_columns[i]
        elif calculate_cosine_similarity("Description", external_coa_columns[i]) >= 50:
            description_column = external_coa_columns[i]
    return name_column, type_column, code_column, description_column


def run_process():
    st.markdown(
        """
        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100px;
            text-align: center;
            font-size: 45px;
            font-weight: bold;
            white-space: nowrap;
        ">
            Jaz COA Import Mapping Tool (TEST)
        </div>
        """,
        unsafe_allow_html=True
    )

    instruction_block = """
    <div style="display: flex; justify-content: center; align-items: center; height: 100%; font-size: 18px; text-align: left;">
        <div style="width: 60%;">
            <h3 style="text-align: center;">Instructions:</h3>
            <ul style="list-style-type: none; padding: 0;">
                <li><strong>The COA external_file should have the full list of accounts you want for the organization account.</strong></li>
                <li><strong>The external_file must have the following columns: jaz_controlled_account, account name, account type.</strong></li>
                <li>Code and description columns are optional, but will be mapped if available.</li>
                <li>Download the organization’s Jaz COA import_file and upload both files in this tool.</li>
                <li>Accounts in the external_file will be matched by AI to those in the import_file. If they match, duplicates will be removed.</li>
                <li>Accounts in the external_file that are not matched will be created as new account rows in the import template.</li>
                <li>Accounts in the import_file that do not have a match on external_file will be set as deleted in the import template.</li>
            </ul>
            <p>If you have any questions or issues, contact <a href="mailto:coa-help@jaz.ai">coa-help@jaz.ai</a>.</p>
        </div>
    </div>
    """

    st.markdown(instruction_block, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.markdown("""
            <div style='text-align: center; margin-bottom: 5px;'>
                <h1 style='font-size: 20px; font-weight: bold;'>Please upload External COA File</h1>
            </div>
        """, unsafe_allow_html=True)
    st.write("")
    external_coa_file = st.file_uploader("", type=["csv"])
    st.write("")
    st.write("")
    st.markdown("""
            <div style='text-align: center; margin-bottom: 5px;'>
                <h1 style='font-size: 20px; font-weight: bold;'>Please upload JAZ COA Import File</h1>
            </div>
        """, unsafe_allow_html=True)
    jaz_coa_file = st.file_uploader("", type=["xlsx"])
    if external_coa_file is not None and jaz_coa_file is not None:
        external_coa_df = pd.read_csv(external_coa_file)
        st.write("external cols", external_coa_df.columns)
        if 'jaz_controlled_account' in external_coa_df.columns:
            external_coa_df.rename(columns={'jaz_controlled_account': 'jaz_sga_name'}, inplace=True)
        if 'jaz_sga_name' not in external_coa_df.columns:
            st.error("""
                Please add a new column “jaz_sga_name” to the external COA file.\t
                In this column, please enter a value from the list below to map to the correct controlled account.
                
                **Controlled Accounts**:
                1. Accounts Payable
                2. Accounts Receivable
                3. Business Bank Account
                4. FX Bank Revaluation (Gains)/Loss
                5. FX Realized Currency (Gains)/Loss
                6. FX Rounding (Gains)/Loss
                7. FX Unrealized Currency (Gains)/Loss
                8. Input GST Receivable
                9. Output VAT Payable
                10. Retained Earnings
                """)
            st.stop()
        name_column, type_column, code_column, description_column = update_external_coa_column_names(external_coa_df)
        if name_column is None and type_column is None:
            st.error("""
            Unable to detect the  Account Name  and  Account Type  column in the External COA file.\t 
            Please update the column name to 'Name' and 'Type' respectively and re-upload the file.
                """)
            st.stop()
        elif name_column is None:
            st.error("""
            Unable to detect the  Account Name  column in the External COA file.\t
            Please update the column name to 'Name' and re-upload the file.
            """)
            st.stop()
        elif type_column is None:
            st.error("""
            Unable to detect the  Account Type  column in the External COA file.\t
            Please update the column name to 'Type' and re-upload the file.
                """)
            st.stop()

        st.write("colop", name_column, type_column, code_column, description_column)
        st.write("before", external_coa_df)
        external_coa_df.rename(columns={name_column: 'Name'}, inplace=True)
        external_coa_df.rename(columns={type_column: 'Type'}, inplace=True)
        code_flag = code_column is not None
        desc_flag = description_column is not None
        if code_flag:
            external_coa_df.rename(columns={code_column: 'Code'}, inplace=True)
        if desc_flag:
            external_coa_df.rename(columns={description_column: 'Description'}, inplace=True)

        st.write("external coa df CP@", external_coa_df)
        external_coa_df = external_coa_df.dropna(subset=['Name', 'Type'])
        jaz_coa_df = pd.read_excel(jaz_coa_file, sheet_name=1)
        jaz_coa_df_columns = jaz_coa_df.columns
        currency_flag = False
        if 'Currency*' in jaz_coa_df_columns:
            currency_flag = True
        column_order = []
        for col in jaz_coa_df_columns:
            if currency_flag:
                if col in COLUMNS_WITH_CURRENCY:
                    column_order.append(col)
            else:
                if col in COLUMNS_WITHOUT_CURRENCY:
                    column_order.append(col)

        st.write("external coa df", external_coa_df, len(external_coa_df))
        for i in range(len(external_coa_df)):
            external_coa_df.at[i, 'Type'] = get_account_type_mapping(external_coa_df.iloc[i]['Type'])
        jaz_coa_map = defaultdict(dict)
        mapped_external_coa_names = set()
        for j in range(len(jaz_coa_df)):
            row = jaz_coa_df.iloc[j]
            account_name = row['Name*']
            account_type = row['Account Type*']
            code = row['Code']
            description = row['Description']
            lock_date = row['Lock Date']
            unique_id = row['Unique ID (do not edit)']
            controlled_account = row['Controlled Account (do not edit)']
            jaz_coa_map[account_name] = {
                "Account Type*": account_type,
                "Name*": account_name,
                "Code": code,
                "Description": description,
                "Lock Date": lock_date,
                "Unique ID (do not edit)": unique_id,
                "Match": False,
                "Status": "DELETE",
                "Controlled Account (do not edit)": controlled_account
            }
            if currency_flag:
                jaz_coa_map[account_name]['Currency*'] = row['Currency*']

        for i in range(len(external_coa_df)):
            row = external_coa_df.iloc[i]
            if row['jaz_sga_name'] == '' or pd.isnull(row['jaz_sga_name']):
                continue
            else:
                for elem, value in jaz_coa_map.items():
                    if (value['Controlled Account (do not edit)'] is not None and
                            value['Controlled Account (do not edit)'] == row['jaz_sga_name']):
                        jaz_coa_map[elem]['Name*'] = row['Name']
                        jaz_coa_map[elem]['Account Type*'] = row['Type']
                        if code_flag:
                            jaz_coa_map[elem]['Code'] = row['Code']
                        if desc_flag:
                            jaz_coa_map[elem]['Description'] = row['Description']
                        jaz_coa_map[elem]['Match'] = True
                        jaz_coa_map[elem]['Status'] = 'ACTIVE'
                        jaz_coa_map[elem]['Match Type'] = 'SGA NAME'
                        mapped_external_coa_names.add(row['Name'])

        jaz_coa_map, mapped_external_coa_names = match_coa_using_gpt(external_coa_df, jaz_coa_df, jaz_coa_map,
                                                                     mapped_external_coa_names, code_flag, desc_flag)

        for p in range(len(external_coa_df)):
            row = external_coa_df.iloc[p]
            if row['Name'] not in mapped_external_coa_names:
                account_name = row['Name']
                account_type = row['Type']
                code = ""
                if code_flag:
                    code = row['Code']
                description = ""
                if desc_flag:
                    description = row['Description']
                lock_date = ""
                unique_id = ""
                controlled_account = ""
                jaz_coa_map[account_name] = {
                    "Account Type*": account_type,
                    "Name*": account_name,
                    "Code": code,
                    "Description": description,
                    "Lock Date": lock_date,
                    "Unique ID (do not edit)": unique_id,
                    "Match": False,
                    "Status": "ACTIVE",
                    "Controlled Account (do not edit)": controlled_account
                }
                if currency_flag:
                    jaz_coa_map[account_name]['Currency*'] = ""

        for key, value in jaz_coa_map.items():
            if value['Controlled Account (do not edit)'] in ACTIVE_ONLY_ACCOUNTS:
                jaz_coa_map[key]['Status'] = 'ACTIVE'

        st.write("JCOAFM", jaz_coa_map)
        final_df = pd.DataFrame.from_dict(jaz_coa_map, orient='index')
        final_df = final_df[column_order]
        final_output_csv = convert_df_to_csv(final_df)
        instructions = """
        <div style="display: flex; justify-content: center; align-items: center; height: 100%; font-size: 18px; text-align: left;">
            <div>
                <h3 style="text-align: center;">Instructions:</h3>
                <p style="margin: 4px 0;"><strong>1. Copy the spreadsheet data and replace the import file entry sheet.</strong></p>
                <p style="margin: 4px 0;"><strong>2. Upload the file to the correct Jaz organization account.</strong></p>
            </div>
        </div>
        """

        # Display the instructions using Streamlit with HTML for center alignment
        st.markdown(instructions, unsafe_allow_html=True)
        st.write("")
        st.write("")
        col1, col2, col3 = st.columns([17, 10, 17])
        with col2:
            st.download_button(
                label="Download File",
                data=final_output_csv,
                file_name='COA_mapped_to_jaz_import.csv',
                mime='text/csv',
            )


if __name__ == '__main__':
    run_process()
