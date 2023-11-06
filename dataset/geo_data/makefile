.DELETE_ON_ERROR:


all: HSall_members.csv HSall_parties.csv

clean:
	rm -f *.csv



HSall_members.csv:
	wget -O 'https://voteview.com/static/data/out/members/HSall_members.csv'

HSall_parties.csv:
	wget -O 'https://voteview.com/static/data/out/parties/HSall_parties.csv'



# Define a list of numbers from 77 to 117
NUMBERS := $(shell seq 77 117)

# Define the base URL
BASE_URL := https://voteview.com/static/data/out/votes/HS

# Define the target file names
FILES := $(addprefix HS, $(addsuffix _votes.csv, $(NUMBERS)))

# Rule to download the files
download_files:
    for number in $(NUMBERS); do \
        wget -O HS$$number_votes.csv $(BASE_URL)$$number_votes.csv; \
    done

# Make the 'download_files' target the default target
.DEFAULT_GOAL := download_files