import pandas as pd
 
 
def read_watspeed_file(path, program_label):
    # row 6 is the actual header in the WatSPEED export format
    df = pd.read_excel(path, header=5)
    df.columns = [
        'Course Title', 'City', 'Province', 'Country',
        'Employer', 'Department', 'Job Title', 'Enrollment Status'
    ]
    df['Program'] = program_label
    df = df.dropna(how='all')
    return df
 
 
def clean_location(df):
    df['City'] = df['City'].str.strip().str.title()
    df['Province'] = df['Province'].str.strip().str.upper()
    df['Country'] = df['Country'].str.strip().str.title()
    return df
 
 
def extract_job_family(title):
    if pd.isna(title):
        return None
 
    cleaned = str(title).strip().title()
 
    terminal_titles = {
        'Director': 'Director',
        'Vp': 'VP',
        'Vice President': 'VP',
        'Svp': 'SVP',
        'Avp': 'AVP',
        'Ceo': 'CEO',
        'Cto': 'CTO',
        'Cfo': 'CFO',
        'President': 'President',
        'Head': 'Head',
        'Owner': 'Owner',
        'Founder': 'Founder',
    }
    for keyword, family in terminal_titles.items():
        if cleaned.startswith(keyword):
            return family
 
    prefixes = [
        'Senior ', 'Junior ', 'Lead ', 'Principal ', 'Staff ',
        'Associate ', 'Acting ', 'Intermediate ', 'Jr ', 'Jr. ',
        'Executive ',
    ]
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
 
    return cleaned.strip()
 
 
def cap_job_families(df, top_n=30):
    top = df['Job Family'].value_counts().nlargest(top_n).index
    df['Job Family'] = df['Job Family'].apply(
        lambda x: x if x in top else 'Other'
    )
    return df
 
 
def load_and_clean(raw_dir='../data/raw'):
    xx1 = read_watspeed_file(f'{raw_dir}/xxxxx1_Enrollment.xlsx', 'xxxxx')
    xx2 = read_watspeed_file(f'{raw_dir}/xxxxx2_Enrollment.xlsx', 'xxxxx')
    ww = read_watspeed_file(f'{raw_dir}/wwwww_Enrollment.xlsx', 'wwwww')
    vv = read_watspeed_file(f'{raw_dir}/vvvvv_Enrollment.xlsx', 'vvvvv')
 
    df = pd.concat([xx1, xx2, ww, vv], ignore_index=True)
    df = clean_location(df)
 
    df['Province'] = df['Province'].fillna('Unknown')
    df['Job Family'] = df['Job Title'].apply(extract_job_family)
    df['Job Family'] = df['Job Family'].fillna('Unknown')
    df = cap_job_families(df)
 
    df['Enrolled_Binary'] = (df['Enrollment Status'] == 'Enrolled').astype(int)
 
    return df