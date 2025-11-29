#include "json_helper.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Very simple JSON parser - assumes keys are unique and values are simple types.
// Not robust for complex or malformed JSON.

static const char* find_key(const char* json, const char* key) {
    char search_key[256];
    snprintf(search_key, sizeof(search_key), "\"%s\":", key);
    return strstr(json, search_key);
}

const char* json_get_string(const char* json, const char* key) {
    const char* key_ptr = find_key(json, key);
    if (!key_ptr) return NULL;

    const char* value_ptr = key_ptr + strlen(key) + 3; // Move past ":"
    if (*value_ptr != '\"') return NULL;

    const char* end_ptr = strchr(value_ptr + 1, '\"');
    if (!end_ptr) return NULL;

    int len = end_ptr - (value_ptr + 1);
    char* result = (char*)malloc(len + 1);
    strncpy(result, value_ptr + 1, len);
    result[len] = '\0';
    return result; // Caller must free this
}

int json_get_int(const char* json, const char* key, int default_val) {
    const char* key_ptr = find_key(json, key);
    if (!key_ptr) return default_val;

    const char* value_ptr = key_ptr + strlen(key) + 2; // Move past ":"
    return atoi(value_ptr);
}

float json_get_float(const char* json, const char* key, float default_val) {
    const char* key_ptr = find_key(json, key);
    if (!key_ptr) return default_val;

    const char* value_ptr = key_ptr + strlen(key) + 2; // Move past ":"
    return atof(value_ptr);
}

int json_get_bool(const char* json, const char* key, int default_val) {
    const char* key_ptr = find_key(json, key);
    if (!key_ptr) return default_val;
    
    const char* value_ptr = key_ptr + strlen(key) + 2; // Move past ":"
    if (strncmp(value_ptr, "true", 4) == 0) {
        return 1;
    }
    if (strncmp(value_ptr, "false", 5) == 0) {
        return 0;
    }
    return default_val;
}
