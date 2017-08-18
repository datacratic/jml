LIBLZ4_SOURCES := \
	xxhash.c \
	lz4.c \
	lz4hc.c \

$(eval $(call library,lz4,$(LIBLZ4_SOURCES),))
$(eval $(call program,lz4cli,lz4,lz4cli.c))
