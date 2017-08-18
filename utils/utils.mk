
LIBUTILS_SOURCES := \
        environment.cc \
        exception_ptr.cc \
        file_functions.cc \
        filter_streams.cc \
        string_functions.cc \
        parse_context.cc \
	configuration.cc \
	csv.cc \
	exc_check.cc \
	exc_assert.cc \
	hex_dump.cc \
	lzma.cc \
	floating_point.cc \
	json_parsing.cc \
	rng.cc \
	hash.cc \
	abort.cc

LIBUTILS_LINK :=	ACE arch lz4 boost_iostreams lzma boost_thread cryptopp

$(eval $(call library,utils,$(LIBUTILS_SOURCES),$(LIBUTILS_LINK)))

# gcc 4.7
$(eval $(call set_compile_option,hash.cc,-fpermissive))

LIBWORKER_TASK_SOURCES := worker_task.cc
LIBWORKER_TASK_LINK    := ACE arch pthread

$(eval $(call library,worker_task,$(LIBWORKER_TASK_SOURCES),$(LIBWORKER_TASK_LINK)))

$(eval $(call include_sub_make,utils_testing,testing))
