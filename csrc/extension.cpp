#include <Python.h>

#if defined(_WIN32)
#define FEATHER_ATTN_EXPORT __declspec(dllexport)
#else
#define FEATHER_ATTN_EXPORT
#endif

extern "C" FEATHER_ATTN_EXPORT PyObject* PyInit__C(void)
{
    static PyModuleDef module = {
        PyModuleDef_HEAD_INIT,
        "_C",
        nullptr,
        -1,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };
    return PyModule_Create(&module);
}
