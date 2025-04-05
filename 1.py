import pandas as pd

def split_excel_by_essay_set(input_file):
    # Read the Excel file
    df = pd.read_excel(input_file)
    
    # Check if 'essay_set' column exists
    if 'essay_set' not in df.columns:
        raise ValueError("Column 'essay_set' not found in the Excel file.")
    
    # Group by 'essay_set' and save each group into a separate file
    for essay_set, group in df.groupby('essay_set'):
        output_filename = f"essay_set_{essay_set}.xlsx"
        group.to_excel(output_filename, index=False)
        print(f"Saved: {output_filename}")

# Call the function with your file
split_excel_by_essay_set("C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//DATASET//asap-aes//training_set_rel3.xlsx")