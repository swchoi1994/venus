#ifndef JSON_HELPER_H
#define JSON_HELPER_H

const char* json_get_string(const char* json, const char* key);
int json_get_int(const char* json, const char* key, int default_val);
float json_get_float(const char* json, const char* key, float default_val);
int json_get_bool(const char* json, const char* key, int default_val);

#endif // JSON_HELPER_H
