MARP     := marp
FLAGS    := --pdf --allow-local-files
SOURCES  := $(wildcard *.md)
OUTPUTS  := $(SOURCES:.md=.pdf)

all: $(OUTPUTS)

%.pdf: %.md
	$(MARP) $(FLAGS) $< -o $@

watch:
	$(MARP) -w $(FLAGS) $(SOURCES)

clean:
	rm -f $(OUTPUTS)

.PHONY: all watch clean
