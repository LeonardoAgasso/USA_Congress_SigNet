.DELETE_ON_ERROR:
.DEFAULT_GOAL := download_votes

N_CONGRESSES := $(shell seq -f "%03g" 77 117)

BASE_URL_VOTES := https://voteview.com/static/data/out/votes/HS
BASE_URL_ROLLCALLS := https://voteview.com/static/data/out/rollcalls/HS

MEMBERS_VOTES_FILES := $(addprefix HS, $(addsuffix _votes.csv, $(N_CONGRESSES)))
CONGRESSIONAL_VOTES_FILES :=  $(addprefix HS, $(addsuffix _rollcalls.csv, $(N_CONGRESSES)))

MEMBERS_VOTES_DIR := ./members_votes/
CONGRESSIONAL_VOTES_DIR := ./congressional_votes/

clean:
	rm -f *.csv

HSall_members.csv:
	wget -O $@ 'https://voteview.com/static/data/out/members/HSall_members.csv'

HSall_parties.csv:
	wget -O $@ 'https://voteview.com/static/data/out/parties/HSall_parties.csv'


download_votes: $(MEMBERS_VOTES_FILES) $(CONGRESSIONAL_VOTES_FILES) HSall_members.csv HSall_parties.csv

download_members_votes: $(MEMBERS_VOTES_FILES)
download_congressional_votes: $(CONGRESSIONAL_VOTES_FILES)


# Rule to download a single file
HS%_votes.csv:
	wget -O $(MEMBERS_VOTES_DIR)$@ "$(BASE_URL_VOTES)$(patsubst HS%,%,$@)"

HS%_rollcalls.csv:
	wget -O $(CONGRESSIONAL_VOTES_DIR)$@ "$(BASE_URL_ROLLCALLS)$(patsubst HS%,%,$@)"