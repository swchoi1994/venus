use std::ffi::{CString, CStr};
use std::os::raw::{c_char, c_int, c_float};
use std::sync::Arc;
use parking_lot::RwLock;
use anyhow::{Result, anyhow};

// FFI bindings to C engine
#[repr(C)]
struct GenerationConfig {
    temperature: c_float,
    top_p: c_float,
    top_k: c_int,
    max_tokens: c_int,
    seed: c_int,
    repetition_penalty: c_float,
    presence_penalty: c_float,
    frequency_penalty: c_float,
}

extern "C" {
    fn create_engine(model_path: *const c_char) -> *mut std::ffi::c_void;
    fn free_engine(engine: *mut std::ffi::c_void);
    fn generate(
        engine: *mut std::ffi::c_void,
        prompt: *const c_char,
        config: *mut GenerationConfig,
    ) -> *mut c_char;
    fn create_tokenizer(tokenizer_path: *const c_char) -> *mut std::ffi::c_void;
    fn free_tokenizer(tokenizer: *mut std::ffi::c_void);
    fn tokenize(
        tokenizer: *mut std::ffi::c_void,
        text: *const c_char,
        n_tokens: *mut c_int,
    ) -> *mut c_int;
}

pub struct VenusEngine {
    engine: *mut std::ffi::c_void,
    tokenizer: *mut std::ffi::c_void,
    model_name: String,
}

unsafe impl Send for VenusEngine {}
unsafe impl Sync for VenusEngine {}

impl VenusEngine {
    pub fn new(model_path: &str) -> Result<Self> {
        let c_path = CString::new(model_path)?;
        
        unsafe {
            let engine = create_engine(c_path.as_ptr());
            if engine.is_null() {
                return Err(anyhow!("Failed to create engine"));
            }
            
            // For now, use the same path for tokenizer
            let tokenizer = create_tokenizer(c_path.as_ptr());
            if tokenizer.is_null() {
                free_engine(engine);
                return Err(anyhow!("Failed to create tokenizer"));
            }
            
            Ok(Self {
                engine,
                tokenizer,
                model_name: model_path.to_string(),
            })
        }
    }
    
    pub fn generate(
        &self,
        prompt: &str,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        max_tokens: i32,
    ) -> Result<String> {
        let c_prompt = CString::new(prompt)?;
        
        let mut config = GenerationConfig {
            temperature,
            top_p,
            top_k,
            max_tokens,
            seed: -1,
            repetition_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        };
        
        unsafe {
            let result = generate(self.engine, c_prompt.as_ptr(), &mut config);
            if result.is_null() {
                return Err(anyhow!("Generation failed"));
            }
            
            let c_str = CStr::from_ptr(result);
            let output = c_str.to_string_lossy().into_owned();
            
            // Free the result string
            libc::free(result as *mut libc::c_void);
            
            Ok(output)
        }
    }
    
    pub fn count_tokens(&self, text: &str) -> Result<i32> {
        let c_text = CString::new(text)?;
        let mut n_tokens: c_int = 0;
        
        unsafe {
            let tokens = tokenize(self.tokenizer, c_text.as_ptr(), &mut n_tokens);
            if !tokens.is_null() {
                libc::free(tokens as *mut libc::c_void);
            }
            
            Ok(n_tokens)
        }
    }
    
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl Drop for VenusEngine {
    fn drop(&mut self) {
        unsafe {
            if !self.engine.is_null() {
                free_engine(self.engine);
            }
            if !self.tokenizer.is_null() {
                free_tokenizer(self.tokenizer);
            }
        }
    }
}

// Engine manager for multiple models
pub struct EngineManager {
    engines: Arc<RwLock<std::collections::HashMap<String, Arc<VenusEngine>>>>,
}

impl EngineManager {
    pub fn new() -> Self {
        Self {
            engines: Arc::new(RwLock::new(std::collections::HashMap::new())),
        }
    }
    
    pub fn load_model(&self, model_name: &str, model_path: &str) -> Result<()> {
        let engine = VenusEngine::new(model_path)?;
        self.engines.write().insert(model_name.to_string(), Arc::new(engine));
        Ok(())
    }
    
    pub fn get_engine(&self, model_name: &str) -> Option<Arc<VenusEngine>> {
        self.engines.read().get(model_name).cloned()
    }
    
    pub fn list_models(&self) -> Vec<String> {
        self.engines.read().keys().cloned().collect()
    }
}