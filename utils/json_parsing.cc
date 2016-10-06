/* json_parsing.cc
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Released under the MIT license.
*/

#include "jml/arch/format.h"
#include "exc_assert.h"
#include "parse_context.h"
#include "json_parsing.h"

using namespace std;


namespace {

const size_t stackBufferMaxSize = 1 << 20; /* 1Mbytes */

inline void
appendUtf8Byte(char * buffer, size_t & pos, uint8_t newChar)
{
    *(buffer + pos) = newChar;
    pos++;
}

inline bool
appendUtf8(char * buffer, size_t & pos, size_t bufferSize, uint32_t cp)
{
    if (cp < 0x80) {
        if (pos >= bufferSize) {
            return false;
        }
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>(cp));
    }
    else if (cp < 0x800) {
        if ((pos + 1) >= bufferSize) {
            return false;
        }
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>((cp >> 6) | 0xc0));
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>((cp & 0x3f) | 0x80));
    }
    else if (cp < 0x10000) {
        if ((pos + 2) >= bufferSize) {
            return false;
        }
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>((cp >> 12) | 0xe0));
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80));
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>((cp & 0x3f) | 0x80));
    }
    else {
        if ((pos + 3) >= bufferSize) {
            return false;
        }
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>((cp >> 18) | 0xf0));
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>(((cp >> 12) & 0x3f) | 0x80));
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>(((cp >> 6) & 0x3f) | 0x80));
        appendUtf8Byte(buffer, pos,
                       static_cast<uint8_t>((cp & 0x3f) | 0x80));
    }

    return true;
}

inline size_t
utf8RequiredBytes(uint32_t cp)
{
    size_t result(1);

    if (cp >= 0x80) {
        if (cp < 0x800) {
            result = 2;
        }
        else if (cp < 0x10000) {
            result = 3;
        }
        else {
            result = 4;
        }
    }

    return result;
}

std::string
expectJsonString(ML::Parse_Context & context, bool acceptUTF8)
{
    skipJsonWhitespace(context);
    context.expect_literal('"');

    char internalBuffer[4096];

    char * buffer = internalBuffer;
    size_t bufferSize = 4096;
    size_t pos = 0;

    // Try multiple times to make it fit
    while (!context.match_literal('"')) {
        unsigned char c = *context++;
        int reqBytes = 1;
        bool unicode(false);
        if (c == '\\') {
            c = *context++;
            switch (c) {
            case 't': c = '\t';  break;
            case 'n': c = '\n';  break;
            case 'r': c = '\r';  break;
            case 'f': c = '\f';  break;
            case 'b': c = '\b';  break;
            case '/': c = '/';   break;
            case '\\':c = '\\';  break;
            case '"': c = '"';   break;
            case 'u': {
                c = context.expect_hex4();
                reqBytes = utf8RequiredBytes(c);
                unicode = true;
                break;
            }
            default:
                context.exception("invalid escaped char");
            }
        }
        if (!acceptUTF8 && c >= 127) {
            context.exception("invalid JSON ASCII string character");
        }
        while ((pos + reqBytes) >= bufferSize) {
            size_t newBufferSize = bufferSize * 8;
            char * newBuffer = new char[newBufferSize];
            std::copy(buffer, buffer + bufferSize, newBuffer);
            if (buffer != internalBuffer)
                delete[] buffer;
            buffer = newBuffer;
            bufferSize = newBufferSize;
        }
        if (unicode) {
            ExcAssert(appendUtf8(buffer, pos, bufferSize, c));
        }
        else {
            /* Input characters that are > 127 are copied verbatim and assumed
               to be utf-8. */
            buffer[pos++] = c;
        }
    }
    
    string result(buffer, buffer + pos);
    if (buffer != internalBuffer)
        delete[] buffer;
    
    return result;
}

inline ssize_t
expectJsonString(ML::Parse_Context & context,
                 char * buffer, size_t maxLength,
                 bool acceptUTF8)
{
    ML::skipJsonWhitespace(context);
    context.expect_literal('"');

    size_t bufferSize = maxLength - 1;
    size_t pos = 0;

    // Try multiple times to make it fit
    while (!context.match_literal('"')) {
        unsigned char c = *context++;
        bool unicode(false);
        if (c == '\\') {
            c = *context++;
            switch (c) {
            case 't': c = '\t'; break;
            case 'n': c = '\n'; break;
            case 'r': c = '\r'; break;
            case 'f': c = '\f'; break;
            case 'b': c = '\b'; break;
            case '/': c = '/'; break;
            case '\\':c = '\\'; break;
            case '"': c = '"'; break;
            case 'u': {
                c = context.expect_hex4();
                unicode = true;
                break;
            }
            default:
                context.exception("invalid escaped char");
            }
        }
        if (!acceptUTF8 && c >= 127) {
            context.exception("invalid JSON ASCII string character");
        }
        if (unicode) {
            if (!appendUtf8(buffer, pos, bufferSize, c)) {
                return -1;
            }
        }
        else {
            /* Input characters that are > 127 are copied verbatim and assumed
               to be utf-8. */
            buffer[pos++] = c;
        }
    }

    buffer[pos] = 0; // null terminator

    return pos;
}

}

namespace ML {

/*****************************************************************************/
/* JSON UTILITIES                                                            */
/*****************************************************************************/

void skipJsonWhitespace(Parse_Context & context)
{
    // Fast-path for the usual case for not EOF and no whitespace
    if (JML_LIKELY(!context.eof())) {
        char c = *context;
        if (c > ' ') {
            return;
        }
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r')
            return;
    }

    while (!context.eof()
           && (context.match_whitespace() || context.match_eol()));
}

char * jsonEscapeCore(const std::string & str, char * p, char * end)
{
    for (unsigned i = 0;  i < str.size();  ++i) {
        if (p + 4 >= end)
            return 0;

        char c = str[i];
        if (c >= ' ' && c < 127 && c != '\"' && c != '\\')
            *p++ = c;
        else {
            *p++ = '\\';
            switch (c) {
            case '\t': *p++ = ('t');  break;
            case '\n': *p++ = ('n');  break;
            case '\r': *p++ = ('r');  break;
            case '\f': *p++ = ('f');  break;
            case '\b': *p++ = ('b');  break;
            case '/':
            case '\\':
            case '\"': *p++ = (c);  break;
            default:
                throw Exception("Invalid character in JSON string: " + str);
            }
        }
    }

    return p;
}

std::string
jsonEscape(const std::string & str)
{
    string result;
    size_t sz = str.size() * 4 + 4;

    result.reserve(sz);
    char * buf = &result[0];
    char * end = buf + sz;

    char * realEnd = jsonEscapeCore(str, buf, end);
    if (!realEnd)
        throw ML::Exception("To fix: logic error in JSON escaping");

    size_t realSz = realEnd - buf;
    result.resize(realSz);

    return result;
}

void jsonEscape(const std::string & str, std::ostream & stream)
{
    size_t sz = str.size() * 4 + 4;

    if (sz < stackBufferMaxSize) {
        char buf[sz];
        char * p = buf, * end = buf + sz;

        p = jsonEscapeCore(str, p, end);

        if (!p)
            throw ML::Exception("To fix: logic error in JSON escaping");

        stream.write(buf, p - buf);
    }
    else {
        string buffer;
        buffer.reserve(sz);
        char * buf = &buffer[0];
        char * p = buf, * end = buf + sz;
        p = jsonEscapeCore(str, p, end);

        if (!p)
            throw ML::Exception("To fix: logic error in JSON escaping");

        stream.write(buf, p - buf);
    }
}

bool matchJsonString(Parse_Context & context, std::string & str)
{
    Parse_Context::Revert_Token token(context);

    skipJsonWhitespace(context);
    if (!context.match_literal('"')) return false;

    std::string result;

    while (!context.match_literal('"')) {
        if (context.eof()) return false;
        unsigned char c = *context++;
        //if (c < 0 || c >= 127)
        //    context.exception("invalid JSON string character");
        if (c != '\\') {
            result.push_back(c);
            continue;
        }
        c = *context++;
        switch (c) {
        case 't': result.push_back('\t');  break;
        case 'n': result.push_back('\n');  break;
        case 'r': result.push_back('\r');  break;
        case 'f': result.push_back('\f');  break;
        case 'b': result.push_back('\b');  break;
        case '/': result.push_back('/');   break;
        case '\\':result.push_back('\\');  break;
        case '"': result.push_back('"');   break;
        case 'u': {
            int code = context.expect_hex4();
            if (code <0 || code > 255)
            {
                return false;
            }
            result.push_back(code);
            break;
        }
        default:
            return false;
        }
    }

    token.ignore();
    str = result;
    return true;
}

std::string
expectJsonStringAscii(Parse_Context & context)
{
    return expectJsonString(context, false);
}

ssize_t
expectJsonStringAscii(Parse_Context & context,
                      char * buf, size_t maxLength)
{
    return expectJsonString(context, buf, maxLength, false);
}

std::string
expectJsonStringUTF8(Parse_Context & context)
{
    return expectJsonString(context, true);
}

ssize_t
expectJsonStringUTF8(Parse_Context & context,
                     char * buf, size_t maxLength)
{
    return expectJsonString(context, buf, maxLength, true);
}

std::string expectJsonStringAsciiPermissive(Parse_Context & context, char sub)
{
    skipJsonWhitespace(context);
    context.expect_literal('"');

    char internalBuffer[4096];

    char * buffer = internalBuffer;
    size_t bufferSize = 4096;
    size_t pos = 0;

    // Try multiple times to make it fit
    while (!context.match_literal('"')) {
        unsigned char c = *context++;
        if (c == '\\') {
            c = *context++;
            switch (c) {
            case 't': c = '\t';  break;
            case 'n': c = '\n';  break;
            case 'r': c = '\r';  break;
            case 'f': c = '\f';  break;
            case 'b': c = '\b';  break;
            case '/': c = '/';   break;
            case '\\':c = '\\';  break;
            case '"': c = '"';   break;
            case 'u': {
                c = context.expect_hex4();
                break;
            }
            default:
                context.exception("invalid escaped char");
            }
        }
        if (c < ' ' || c >= 127)
            c = sub;
        if (pos == bufferSize) {
            size_t newBufferSize = bufferSize * 8;
            char * newBuffer = new char[newBufferSize];
            std::copy(buffer, buffer + bufferSize, newBuffer);
            if (buffer != internalBuffer)
                delete[] buffer;
            buffer = newBuffer;
            bufferSize = newBufferSize;
        }
        buffer[pos++] = c;
    }

    string result(buffer, buffer + pos);
    if (buffer != internalBuffer)
        delete[] buffer;
    
    return result;
}

bool
matchJsonNull(Parse_Context & context)
{
    skipJsonWhitespace(context);
    return context.match_literal("null");
}

void
expectJsonArray(Parse_Context & context,
                const std::function<void (int, Parse_Context &)> & onEntry)
{
    skipJsonWhitespace(context);

    if (context.match_literal("null"))
        return;

    context.expect_literal('[');
    skipJsonWhitespace(context);
    if (context.match_literal(']')) return;

    for (int i = 0;  ; ++i) {
        skipJsonWhitespace(context);

        onEntry(i, context);

        skipJsonWhitespace(context);

        if (!context.match_literal(',')) break;
    }

    skipJsonWhitespace(context);
    context.expect_literal(']');
}

void
expectJsonObject(Parse_Context & context,
                 const std::function<void (const std::string &, Parse_Context &)> & onEntry)
{
    skipJsonWhitespace(context);

    if (context.match_literal("null"))
        return;

    context.expect_literal('{');

    skipJsonWhitespace(context);

    if (context.match_literal('}')) return;

    for (;;) {
        skipJsonWhitespace(context);

        string key = expectJsonStringAscii(context);

        skipJsonWhitespace(context);

        context.expect_literal(':');

        skipJsonWhitespace(context);

        onEntry(key, context);

        skipJsonWhitespace(context);

        if (!context.match_literal(',')) break;
    }

    skipJsonWhitespace(context);
    context.expect_literal('}');
}

void
expectJsonObjectAscii(Parse_Context & context,
                      const std::function<void (const char *, Parse_Context &)> & onEntry)
{
    skipJsonWhitespace(context);

    if (context.match_literal("null"))
        return;

    context.expect_literal('{');

    skipJsonWhitespace(context);

    if (context.match_literal('}')) return;

    for (;;) {
        skipJsonWhitespace(context);

        char keyBuffer[1024];

        ssize_t done = expectJsonStringAscii(context, keyBuffer, 1024);
        if (done == -1)
            context.exception("JSON key is too long");

        skipJsonWhitespace(context);

        context.expect_literal(':');

        skipJsonWhitespace(context);

        onEntry(keyBuffer, context);

        skipJsonWhitespace(context);

        if (!context.match_literal(',')) break;
    }

    skipJsonWhitespace(context);
    context.expect_literal('}');
}

bool
matchJsonObject(Parse_Context & context,
                const std::function<bool (const std::string &, Parse_Context &)> & onEntry)
{
    skipJsonWhitespace(context);

    if (context.match_literal("null"))
        return true;

    if (!context.match_literal('{')) return false;
    skipJsonWhitespace(context);
    if (context.match_literal('}')) return true;

    for (;;) {
        skipJsonWhitespace(context);

        string key = expectJsonStringAscii(context);

        skipJsonWhitespace(context);
        if (!context.match_literal(':')) return false;
        skipJsonWhitespace(context);

        if (!onEntry(key, context)) return false;

        skipJsonWhitespace(context);

        if (!context.match_literal(',')) break;
    }

    skipJsonWhitespace(context);
    if (!context.match_literal('}')) return false;

    return true;
}

JsonNumber expectJsonNumber(Parse_Context & context)
{
    JsonNumber result;

    std::string number;
    number.reserve(32);

    bool negative = false;
    bool doublePrecision = false;

    if (context.match_literal('-')) {
        number += '-';
        negative = true;
    }

    // EXTENSION: accept NaN and positive or negative infinity
    if (context.match_literal('N')) {
        context.expect_literal("aN");
        result.fp = negative ? -NAN : NAN;
        result.type = JsonNumber::FLOATING_POINT;
        return result;
    }
    else if (context.match_literal('n')) {
        context.expect_literal("an");
        result.fp = negative ? -NAN : NAN;
        result.type = JsonNumber::FLOATING_POINT;
        return result;
    }
    else if (context.match_literal('I') || context.match_literal('i')) {
        context.expect_literal("nf");
        result.fp = negative ? -INFINITY : INFINITY;
        result.type = JsonNumber::FLOATING_POINT;
        return result;
    }

    while (context && isdigit(*context)) {
        number += *context++;
    }

    if (context.match_literal('.')) {
        doublePrecision = true;
        number += '.';

        while (context && isdigit(*context)) {
            number += *context++;
        }
    }

    char sci = context ? *context : '\0';
    if (sci == 'e' || sci == 'E') {
        doublePrecision = true;
        number += *context++;

        char sign = context ? *context : '\0';
        if (sign == '+' || sign == '-') {
            number += *context++;
        }

        while (context && isdigit(*context)) {
            number += *context++;
        }
    }

    try {
        JML_TRACE_EXCEPTIONS(false);
        if (number.empty())
            context.exception("expected number");

        if (doublePrecision) {
            char * endptr = 0;
            errno = 0;
            result.fp = strtod(number.c_str(), &endptr);
            if (errno || endptr != number.c_str() + number.length())
                context.exception(ML::format("failed to convert '%s' to long long",
                                             number.c_str()));
            result.type = JsonNumber::FLOATING_POINT;
        } else if (negative) {
            char * endptr = 0;
            errno = 0;
            result.sgn = strtol(number.c_str(), &endptr, 10);
            if (errno || endptr != number.c_str() + number.length())
                context.exception(ML::format("failed to convert '%s' to long long",
                                             number.c_str()));
            result.type = JsonNumber::SIGNED_INT;
        } else {
            char * endptr = 0;
            errno = 0;
            result.uns = strtoull(number.c_str(), &endptr, 10);
            if (errno || endptr != number.c_str() + number.length())
                context.exception(ML::format("failed to convert '%s' to unsigned long long",
                                             number.c_str()));
            result.type = JsonNumber::UNSIGNED_INT;
        }
    } catch (const std::exception & exc) {
        context.exception("expected number");
    }

    return result;
}

} // namespace ML
