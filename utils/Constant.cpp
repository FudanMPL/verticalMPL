#include "Constant.h"

u64 Constant::Clock::global_clock[101] = {0};

void Constant::Clock::print_clock(int id)
{
    DBGprint("clock: %d %f\n", id, global_clock[id] * 1.0 * microseconds::period::num / microseconds::period::den);
}

double Constant::Clock::get_clock(int id)
{
    return global_clock[id] * 1.0 * microseconds::period::num / microseconds::period::den;
}

void Constant::Util::int_to_char(char *&p, int u)
{
    *p++ = u & 0xff;
    *p++ = u >> 8 & 0xff;
    *p++ = u >> 16 & 0xff;
    *p++ = u >> 24 & 0xff;
}

void Constant::Util::u64_to_char(char *&p, u64 u)
{
    *p++ = u & 0xff;
    *p++ = u >> 8 & 0xff;
    *p++ = u >> 16 & 0xff;
    *p++ = u >> 24 & 0xff;
    *p++ = u >> 32 & 0xff;
    *p++ = u >> 40 & 0xff;
    *p++ = u >> 48 & 0xff;
    *p++ = u >> 56 & 0xff;
}

int Constant::Util::char_to_int(char *&p)
{
    int ret = 0;
    ret = *p++ & 0xff;
    ret |= (*p++ & 0xff) << 8;
    ret |= (*p++ & 0xff) << 16;
    ret |= (*p++ & 0xff) << 24;
    return ret;
}

u64 Constant::Util::char_to_u64(char *&p)
{
    u64 ret = 0;
    ret = (u64)(*p++ & 0xff);
    ret |= (u64)(*p++ & 0xff) << 8;
    ret |= (u64)(*p++ & 0xff) << 16;
    ret |= (u64)(*p++ & 0xff) << 24;
    ret |= (u64)(*p++ & 0xff) << 32;
    ret |= (u64)(*p++ & 0xff) << 40;
    ret |= (u64)(*p++ & 0xff) << 48;
    ret |= (u64)(*p++ & 0xff) << 56;
    return ret;
}

void Constant::Util::int_to_header(char *p, int u)
{
    *p++ = u & 0xff;
    *p++ = u >> 8 & 0xff;
    *p++ = u >> 16 & 0xff;
    *p++ = u >> 24 & 0xff;
}

int Constant::Util::header_to_int(char *p)
{
    int ret = 0;
    ret = *p++ & 0xff;
    ret |= (*p++ & 0xff) << 8;
    ret |= (*p++ & 0xff) << 16;
    ret |= (*p++ & 0xff) << 24;
    return ret;
}

u64 Constant::Util::double_to_u64(double x)
{
    return static_cast<u64>((ll)(x * UINT64_MASK));
}

double Constant::Util::u64_to_double(u64 u)
{
    return (ll)u / (double)UINT64_MASK;
}

double Constant::Util::char_to_double(char *&p)
{
    return strtod(p, NULL);
}

int Constant::Util::getint(char *&p)
{
    while (!isdigit(*p))
        p++;
    int ret = 0;
    while (isdigit(*p))
    {
        ret = 10 * ret + *p - '0';
        p++;
    }
    return ret;
}

u64 Constant::Util::getu64(char *&p)
{
    while (!isdigit(*p))
        p++;
    u64 ret = 0;
    while (isdigit(*p))
    {
        ret = 10 * ret + *p - '0';
        p++;
    }
    return ret;
}

u64 Constant::Util::random_u64()
{
    u64 ra = (u64)(abs(rand()));
    u64 rb = (u64)(abs(rand()));
    ra <<= (sizeof(int) * 8);
    return (u64)(ra | rb);
}

u64 Constant::Util::multiply(u64 a, u64 b)
{
    auto la = (ll)a;
    auto lb = (ll)b;
    ll product = (la * lb) / (ll)UINT64_MASK;
    return static_cast<u64>(product);
}

u64 Constant::Util::truncate(u64 x)
{
    return static_cast<u64>((ll)x / (ll)UINT64_MASK);
}

u64 Constant::Util::divide(u64 a, int b)
{
    return static_cast<uint64_t>((ll)a / (double)b);
}

u64 Constant::Util::divide(u64 a, u64 b)
{
    double a1 = Constant::Util::u64_to_double(a);
    double b1 = Constant::Util::u64_to_double(b);
    return Constant::Util::double_to_u64(a1 / b1);
}
