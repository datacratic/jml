LZ4CLI_SOURCES = lz4cli.c lz4io.c bench.c datagen.c
$(eval $(call program,lz4cli,lz4,$(LZ4CLI_SOURCES)))
$(eval $(call set_compile_option,$(LZ4CLI_SOURCES),-Ijml/lz4))
