LIBLZ4_SOURCES := \
	xxhash.c \
	lz4.c \
	lz4frame.c \
	lz4hc.c

$(eval $(call library,lz4,$(LIBLZ4_SOURCES),))
$(eval $(call include_sub_makes,lz4cli))
