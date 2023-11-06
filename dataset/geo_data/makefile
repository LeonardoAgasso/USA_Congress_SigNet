.DELETE_ON_ERROR:
.DEFAULT_GOAL := download_votes

N_CONGRESSES := $(shell seq -f "%03g" 77 117)

BASE_URL_VOTES := https://voteview.com/static/data/out/votes/HS
BASE_URL_ROLLCALLS := https://voteview.com/static/data/out/rollcalls/HS

MEMBERS_VOTES_FILES := $(addprefix HS, $(addsuffix _votes.tsv, $(N_CONGRESSES)))
CONGRESSIONAL_VOTES_FILES :=  $(addprefix HS, $(addsuffix _rollcalls.tsv, $(N_CONGRESSES)))

MVDIR := ./members_votes/
CVDIR := ./congressional_votes/

clean:
	rm -f *.csv

HSall_members.csv:
	wget -O $@ 'https://voteview.com/static/data/out/members/HSall_members.csv'

HSall_parties.csv:
	wget -O $@ 'https://voteview.com/static/data/out/parties/HSall_parties.csv'


download_votes: $(MEMBERS_VOTES_FILES) $(CONGRESSIONAL_VOTES_FILES) HSall_members.csv HSall_parties.csv

download_members_votes: $(MEMBERS_VOTES_FILES)
download_congressional_votes: $(CONGRESSIONAL_VOTES_FILES)
merge_members_congressional_votes:


# Single files
HS%_votes.csv:
	wget -O $(MVDIR)$@ "$(BASE_URL_VOTES)$(patsubst HS%,%,$@)"

HS%_rollcalls.csv:
	wget -O $(CVDIR)$@ "$(BASE_URL_ROLLCALLS)$(patsubst HS%,%,$@)"

HS%_votes.tsv: HS%_votes.csv
	csv2tsv < $(MVDIR)$< > $(MVDIR)$@;
	rm $(MVDIR)$<

HS%_rollcalls.tsv: HS%_rollcalls.csv
	csv2tsv < $(CVDIR)$< > $(MVDIR)$@
	rm $(CVDIR)$<

