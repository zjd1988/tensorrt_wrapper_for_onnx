/********************************************
// Filename: non_copyable.hpp
// Created by zjd1988 on 2024/12/27
// Description:

********************************************/
#pragma once

namespace TENSORRT_WRAPPER
{

    /** protocol class. used to delete assignment operator. */
    class NonCopyable
    {
    public:
        NonCopyable() = default;
        NonCopyable(const NonCopyable&) = delete;
        NonCopyable(const NonCopyable&&) = delete;
        NonCopyable& operator=(const NonCopyable&) = delete;
        NonCopyable& operator=(const NonCopyable&&) = delete;
    };

} // namespace TENSORRT_WRAPPER