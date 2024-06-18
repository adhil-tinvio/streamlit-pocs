import streamlit as st
import pandas as pd
import requests
import concurrent.futures
from collections import defaultdict

API_KEY = st.secrets["API_KEY"]


def sga_prompt_generator(chart_of_accounts):
    sga_prompt = """
    You are a skilled financial analyst.
    Match the given account names with the most suitable account from the Chart of Accounts given to you.
    Below is the list of available Chart of Accounts in the format account type - account name:
    the input will also be similiar account type and name
    """ + "\n- " + "\n- ".join(chart_of_accounts) + "\n"

    sga_prompt += """
    1) If there is no close match, name it 'No Suitable Match'.
    2) The matches must be 1:1, meaning each account name from the list must be paired uniquely with one account from the COA and vice versa. This is very important.
    3) If there are 15 bank accounts given in a batch, you must return exactly 15 mapped account types (meaning 15 values returned, even if all 15 are no suitable match), Including No Suitable Match. This is so that the format will not get messed up. 
    4) Please do not give them index numbers at all.
    5) Make sure the return list length is exactly the same as the input size length (VERY IMPORTANT PLEASE MAKE SURE FOR EVERY BATCH)
    6) Please do not have empty lines in your return. The results should all be in the next line IMPORTANT
    7) the account types when matched should not be conflciting
    """

    return sga_prompt


def recommend_sga_match(coa_names, account_names,account_types, batch_size=15):
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    results = [None] * len(account_names)
    sga_prompt = sga_prompt_generator(coa_names)
    st.write("COa nems", coa_names)
    st.write("SGA prompt", sga_prompt)
    st.write("account details", account_names,account_types)

    def process_batch(start_index):
        end_index = min(start_index + batch_size, len(account_names))
        #batch = account_details[start_index:end_index]
        c_an=account_names[start_index:end_index]
        c_at=account_types[start_index:end_index]
        messages = [{'role': 'system', 'content': sga_prompt}]
        for t in range(len(c_an)):
            messages.append({'role': 'user', 'content': f"Account Type: {c_at[t]} - Account Name:{c_an[t]}"})

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
            batch_results = [line.strip() for line in content.split('\n') if line.strip()]

            if len(batch_results) != len(c_an):
                print("batch: ", c_an)
                print("batch results: ", batch_results)
                raise ValueError(f"Expected {len(c_an)} results, but got {len(batch_results)}")
        except (requests.exceptions.RequestException, ValueError, IndexError) as e:
            print(f"Error processing batch: {e}")
            batch_results = ["Error in classification"] * len(c_an)

        results[start_index:end_index] = batch_results

    with concurrent.futures.ThreadPoolExecutor() as executor:
        indices = range(0, len(account_names), batch_size)
        executor.map(process_batch, indices)

    return results


def match_coa_using_gpt(external_coa, jaz_coa, chart_of_accounts_map, mapped_coa_names):
    st.write(external_coa, "COA")
    st.write(external_coa.columns, "COLS")
    unmapped_external_coa=external_coa[~(external_coa['*Name'].isin(mapped_coa_names))]
    external_coa_account_names = [name for name in unmapped_external_coa]
    st.write(external_coa_account_names, "ACNAMES")
    st.write("jaz template", jaz_coa)
    jazz_an = []
    jazz_at= []
    st.write("jaz_ans", jaz_coa['Name*'].tolist())
    for i in range(len(jaz_coa)):
        name=jaz_coa.iloc[i]['Name*']
        ac_type=jaz_coa.iloc[i]['Account Type*']
        if chart_of_accounts_map[name]['Match']:
            continue
        else:
            jazz_an.append(name)
            jazz_at.append(ac_type)
    st.write("jax an ", jazz_an)
    #combined_accounts = [f"{name} " for name in external_coa_account_names]
    ext_coa_account_names = unmapped_external_coa['*Name'].tolist()
    ext_coa_account_types = unmapped_external_coa['*Type'].tolist()
    combined_accounts = [f"{name} - {acnt_type}" for name, acnt_type in zip(ext_coa_account_names, ext_coa_account_types)]
    jaz_accounts = [f"{name} - {ac_type}" for name,ac_type in zip(jazz_an, jazz_at)]
    st.write("JAN", jaz_accounts)
    sga_matches = recommend_sga_match(jaz_accounts, ext_coa_account_names,ext_coa_account_types, 15)
    st.write("SGA_matches", sga_matches)
    st.write("len sga matches", len(sga_matches), len(external_coa_account_names), len(combined_accounts))
    for i in range(len(sga_matches)):
        if sga_matches[i] != 'No Suitable Match':
            jaz_coa_name = sga_matches[i]
            ext_coa_name = external_coa_account_names[i]
            mapped_coa_names.add(ext_coa_name)
            filtered_df = external_coa[(ext_coa_name['*Name'] == ext_coa_name)]
            st.write("filtered df length", len(filtered_df), ext_coa_name, filtered_df)
            rowz = filtered_df[0]
            chart_of_accounts_map[jaz_coa_name]['Code'] = rowz['*Code']
            chart_of_accounts_map[jaz_coa_name]['Description'] = rowz['Description']
            chart_of_accounts_map[jaz_coa_name]['Match'] = True
            chart_of_accounts_map[jaz_coa_name]['Status'] = 'ACTIVE'
    return chart_of_accounts_map, mapped_coa_names


def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')


st.title('Jaz COA Import Mapping Tool (SG-EN)')
external_coa_file = st.file_uploader("Choose the external coa file", type=["csv"])
jaz_coa_file = st.file_uploader("Choose the jaz coa import file", type=["xlsx"])
if external_coa_file is not None and jaz_coa_file is not None:
    #st.write("print_sga prompt", sga_prompt)
    external_coa_data = pd.read_csv(external_coa_file)
    jaz_coa_data = pd.read_excel(jaz_coa_file, sheet_name=1)
    coa_map = defaultdict(dict)
    mapped_external_coa_names = set()
    for j in range(len(jaz_coa_data)):
        row = jaz_coa_data.iloc[j]
        account_name = row['Name*']
        account_type = row['Account Type*']
        code = row['Code']
        description = row['Description']
        lock_date = row['Lock Date']
        unique_id = row['Unique ID (do not edit)']
        coa_map[account_name] = {
            "Account Type*": account_type,
            "Name*": account_name,
            "Code": code,
            "Description": description,
            "Lock Date": lock_date,
            "Unique ID (do not edit)": unique_id,
            "Match": False,
            "Status": "INACTIVE"
        }

    for i in range(len(external_coa_data)):
        row = external_coa_data.iloc[i]
        if row['jaz_sga_name'] == '' or pd.isnull(row['jaz_sga_name']):
            continue
        else:
            if row['jaz_sga_name'] in coa_map:
                coa_map[row['jaz_sga_name']]['Code'] = row['*Code']
                coa_map[row['jaz_sga_name']]['Description'] = row['Description']
                coa_map[row['jaz_sga_name']]['Match'] = True
                coa_map[row['jaz_sga_name']]['Status'] = 'ACTIVE'
                mapped_external_coa_names.add(row['*Name'])
    st.write("GOAT", coa_map)
    coa_map, mapped_external_coa_names = match_coa_using_gpt(external_coa_data, jaz_coa_data, coa_map,
                                                             mapped_external_coa_names)
    st.write("Processed Data", coa_map)
    for p in range(len(external_coa_data)):
        row = external_coa_data.iloc[p]
        if row['*Name'] not in mapped_external_coa_names:
            account_name = row['*Name']
            account_type = row['*Type']
            code = row['*Code']
            description = row['Description']
            lock_date = row['Lock Date']
            unique_id = ""
            coa_map[account_name] = {
                "Account Type*": account_type,
                "Name*": account_name,
                "Code": code,
                "Description": description,
                "Lock Date": lock_date,
                "Unique ID (do not edit)": unique_id,
                "Match": False,
                "Status": "ACTIVE"
            }
    f_df = pd.DataFrame.from_dict(coa_map, orient='index')
    # Reset the index to move the outer dictionary keys to a column
    f_df.reset_index(drop=True, inplace=True)
    csv = convert_df_to_csv(f_df)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='mapped_coa.csv',
        mime='text/csv',
    )
