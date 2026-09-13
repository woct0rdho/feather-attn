#include <Python.h>

extern "C" PyObject* PyInit__C(void)
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
