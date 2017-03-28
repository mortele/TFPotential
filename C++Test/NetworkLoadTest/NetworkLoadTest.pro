TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

INCLUDEPATH += /usr/local/include
INCLUDEPATH += /usr/local/Cellar

LIBS += -L/usr/local/lib
LIBS += -larmadillo -lblas -llapack

SOURCES += main.cpp \
    neuralnetwork.cpp \
    activationfunction.cpp

HEADERS += \
    neuralnetwork.h \
    activationfunction.h
