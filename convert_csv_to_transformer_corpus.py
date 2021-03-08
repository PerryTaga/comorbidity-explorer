# Utility to turn conditions csv into a text file of chronological conditions, one line per patient


INPUT_CSV_DIRECTORY = "./data/conditions-small.csv"
OUTPUT_CORPUS_DIRECTORY = "./corpus/corpus-small.txt"



# ----------------------------------------- Helper Functions -----------------------------------------
# returns a single line string in the format:
# this_is_item_1,this_is_item_2,etc
def listToEncodedString(input_list):

	output_string = ""

	for item in input_list:
		output_string = output_string + item.replace(" ", "_") + " "

	# remove the final comma
	output_string = output_string[:-1]

	return output_string
# ----------------------------------------------------------------------------------------------------



import pandas as pd

# UPDATE DATE FILE NAME AS NEEDED
raw_data = pd.read_csv(INPUT_CSV_DIRECTORY)

# sort by condition start date
time_sorted_raw_data = raw_data.sort_values(by=['START'])

# grab unique patient IDs from the dataframe
list_of_patients = time_sorted_raw_data.PATIENT.unique().tolist()

# compile and encode patient hisotries, based on ID
patient_history_strings = []
for patient_id in list_of_patients:
	patient_history_df = time_sorted_raw_data[time_sorted_raw_data['PATIENT']==patient_id]
	# removes duplicate conditions for this prediction model
	patient_history_list = patient_history_df.DESCRIPTION.unique().tolist()
	patient_history_strings.append(listToEncodedString(patient_history_list) + "\n")

# remove the final newline
patient_history_strings[-1] = patient_history_strings[-1][:-1]

# write histories to corpus
output_file = open(OUTPUT_CORPUS_DIRECTORY, "w")
output_file.writelines(patient_history_strings)
output_file.close()

print("Successfully generated corpus at: " + OUTPUT_CORPUS_DIRECTORY)