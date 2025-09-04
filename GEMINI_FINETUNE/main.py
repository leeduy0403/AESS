from evaluation.evaluator import evaluate_essays

if __name__ == "__main__":
    sets = range(1, 9)
    for i in sets:
        evaluate_essays(
            f"C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET{i}//essay_set_{i}_description.docx",
            f"C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//Submission//SET{i}//essay_set_{i}_test.xlsx",
            f"C://Users//Admin//OneDrive//DACN+DATN//AI MODEL//AES//OUTPUT//TEST_FLOW//set{i}.xlsx"
        )