/* json_parsing.h                                                  -*- C++ -*-
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Released under the MIT license.

   Functionality to ease parsing of JSON from within a parse_function.
*/

#pragma once

#include <string>
#include <functional>


namespace ML {

/* Forward declaration */
struct Parse_Context;


/*****************************************************************************/
/* JSON UTILITIES                                                            */
/*****************************************************************************/

std::string jsonEscape(const std::string & str);

void jsonEscape(const std::string & str, std::ostream & out);

/*
 * If non-ascii characters are found an exception is thrown
 */
std::string expectJsonStringAscii(Parse_Context & context);

/*
 * If non-ascii characters are found an exception is thrown.
 * Output goes into the given buffer, of the given maximum length.
 * If it doesn't fit, then return zero.
 */
ssize_t expectJsonStringAscii(Parse_Context & context, char * buf,
                              size_t maxLength);

/*
 * if non-ascii characters are found we replace them by an ascii character that is supplied
 */
std::string expectJsonStringAsciiPermissive(Parse_Context & context, char c);

/*
 * Decode JSON Unicode strings using utf-8 encoding. The input supports both
 * characters encoded in the form and plain utf-8 characters.
 */
std::string expectJsonStringUTF8(Parse_Context & context);
ssize_t expectJsonStringUTF8(Parse_Context & context,
                             char * buf, size_t maxLength);

/*
 * If non-ascii characters are found an exception is thrown.
 * Output goes into the given buffer, of the given maximum length.
 * If it doesn't fit, then return zero.
 */
ssize_t expectJsonStringAscii(Parse_Context & context, char * buf,
                             size_t maxLength);

bool matchJsonString(Parse_Context & context, std::string & str);

bool matchJsonNull(Parse_Context & context);

void
expectJsonArray(Parse_Context & context,
                const std::function<void (int, Parse_Context &)> & onEntry);

void
expectJsonObject(Parse_Context & context,
                 const std::function<void (const std::string &, Parse_Context &)> & onEntry);

/** Expect a Json object and call the given callback.  The keys are assumed
    to be ASCII which means no embedded nulls, and so the key can be passed
    as a const char *.
*/
void
expectJsonObjectAscii(Parse_Context & context,
                      const std::function<void (const char *, Parse_Context &)> & onEntry);

bool
matchJsonObject(Parse_Context & context,
                const std::function<bool (const std::string &, Parse_Context &)> & onEntry);

void skipJsonWhitespace(Parse_Context & context);

bool expectJsonBool(Parse_Context & context);

/** Representation of a numeric value in JSON.  It's designed to allow
    it to be stored the same way it was written (as an integer versus
    floating point, signed vs unsigned) without losing precision.
*/
struct JsonNumber {
    enum Type {
        NONE,
        UNSIGNED_INT,
        SIGNED_INT,
        FLOATING_POINT
    } type;

    union {
        unsigned long long uns;
        long long sgn;
        double fp;
    };    
};

/** Expect a JSON number.  This function is written in this strange way
    because JsonCPP is not a require dependency of jml, but the function
    needs to be out-of-line.
*/
JsonNumber expectJsonNumber(Parse_Context & context);

/** Match a JSON number. */
bool matchJsonNumber(Parse_Context & context, JsonNumber & num);

} // namespace ML

