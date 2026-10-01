//! Reader for the Arrow C data and C stream interfaces.
//!
//! upstream: `include/LightGBM/arrow.h` (C structs) and
//! `src/arrow/array.hpp` (`ArrowChunkedArray`, its `View`, `Visitor` and
//! `Iterator`). Values are cast to the requested output type as upstream
//! `static_cast`s them; nulls become NaN for floating-point outputs and 0
//! otherwise. Supported value types are the integer types, float, double and
//! bool; a struct (record batch) holds one field per feature column.

use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr;

use crate::error::{LgbmError, Result};

/// `struct ArrowSchema` of the Arrow C data interface.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowSchema {
    pub format: *const c_char,
    pub name: *const c_char,
    pub metadata: *const c_char,
    pub flags: i64,
    pub n_children: i64,
    pub children: *mut *mut ArrowSchema,
    pub dictionary: *mut ArrowSchema,
    pub release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    pub private_data: *mut c_void,
}

/// `struct ArrowArray` of the Arrow C data interface.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowArray {
    pub length: i64,
    pub null_count: i64,
    pub offset: i64,
    pub n_buffers: i64,
    pub n_children: i64,
    pub buffers: *mut *const c_void,
    pub children: *mut *mut ArrowArray,
    pub dictionary: *mut ArrowArray,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    pub private_data: *mut c_void,
}

/// `struct ArrowArrayStream` of the Arrow C stream interface.
#[repr(C)]
#[derive(Debug)]
pub struct ArrowArrayStream {
    pub get_schema: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowSchema) -> c_int>,
    pub get_next: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowArray) -> c_int>,
    pub get_last_error: Option<unsafe extern "C" fn(*mut ArrowArrayStream) -> *const c_char>,
    pub release: Option<unsafe extern "C" fn(*mut ArrowArrayStream)>,
    pub private_data: *mut c_void,
}

impl ArrowSchema {
    fn empty() -> Self {
        Self {
            format: ptr::null(),
            name: ptr::null(),
            metadata: ptr::null(),
            flags: 0,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

impl ArrowArray {
    fn empty() -> Self {
        Self {
            length: 0,
            null_count: 0,
            offset: 0,
            n_buffers: 0,
            n_children: 0,
            buffers: ptr::null_mut(),
            children: ptr::null_mut(),
            dictionary: ptr::null_mut(),
            release: None,
            private_data: ptr::null_mut(),
        }
    }
}

/// Logical type of an array (the subset of nanoarrow's `ArrowType` that matters here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ArrowType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float,
    Double,
    Bool,
    Struct,
    Other(&'static str),
}

impl ArrowType {
    /// upstream: nanoarrow `ArrowSchemaViewInit` (format string -> type) and `ArrowTypeString`.
    fn of(schema: &ArrowSchema) -> Result<Self> {
        if schema.format.is_null() {
            return Err(LgbmError::InvalidData(
                "Failed to initialize ArrowSchemaView: Expected non-NULL schema->format".into(),
            ));
        }
        if !schema.dictionary.is_null() {
            return Ok(Self::Other("dictionary"));
        }
        let format = unsafe { CStr::from_ptr(schema.format) }.to_bytes();
        Ok(match format {
            b"c" => Self::Int8,
            b"s" => Self::Int16,
            b"i" => Self::Int32,
            b"l" => Self::Int64,
            b"C" => Self::UInt8,
            b"S" => Self::UInt16,
            b"I" => Self::UInt32,
            b"L" => Self::UInt64,
            b"f" => Self::Float,
            b"g" => Self::Double,
            b"b" => Self::Bool,
            b"+s" => Self::Struct,
            b"n" => Self::Other("na"),
            b"e" => Self::Other("half_float"),
            b"u" => Self::Other("string"),
            b"U" => Self::Other("large_string"),
            b"vu" => Self::Other("string_view"),
            b"z" => Self::Other("binary"),
            b"Z" => Self::Other("large_binary"),
            b"vz" => Self::Other("binary_view"),
            b"tdD" => Self::Other("date32"),
            b"tdm" => Self::Other("date64"),
            b"+l" => Self::Other("list"),
            b"+L" => Self::Other("large_list"),
            b"+vl" => Self::Other("list_view"),
            b"+vL" => Self::Other("large_list_view"),
            b"+m" => Self::Other("map"),
            b"+r" => Self::Other("run_end_encoded"),
            f if f.starts_with(b"ts") => Self::Other("timestamp"),
            f if f.starts_with(b"tts") || f.starts_with(b"ttm") => Self::Other("time32"),
            f if f.starts_with(b"ttu") || f.starts_with(b"ttn") => Self::Other("time64"),
            f if f.starts_with(b"tD") => Self::Other("duration"),
            b"tiM" => Self::Other("interval_months"),
            b"tiD" => Self::Other("interval_day_time"),
            b"tin" => Self::Other("interval_month_day_nano"),
            f if f.starts_with(b"w:") => Self::Other("fixed_size_binary"),
            f if f.starts_with(b"+w:") => Self::Other("fixed_size_list"),
            f if f.starts_with(b"+us:") => Self::Other("sparse_union"),
            f if f.starts_with(b"+ud:") => Self::Other("dense_union"),
            f if f.starts_with(b"d:") => {
                let s = String::from_utf8_lossy(f);
                match s.split(',').nth(2) {
                    Some("32") => Self::Other("decimal32"),
                    Some("64") => Self::Other("decimal64"),
                    Some("256") => Self::Other("decimal256"),
                    _ => Self::Other("decimal128"),
                }
            }
            f => {
                return Err(LgbmError::InvalidData(format!(
                    "Failed to initialize ArrowSchemaView: Error parsing schema->format: Unknown format: '{}'",
                    String::from_utf8_lossy(f)
                )));
            }
        })
    }

    fn name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int16 => "int16",
            Self::Int32 => "int32",
            Self::Int64 => "int64",
            Self::UInt8 => "uint8",
            Self::UInt16 => "uint16",
            Self::UInt32 => "uint32",
            Self::UInt64 => "uint64",
            Self::Float => "float",
            Self::Double => "double",
            Self::Bool => "bool",
            Self::Struct => "struct",
            Self::Other(s) => s,
        }
    }
}

/// Output types of [`ArrowChunkedArray`] value accessors (`OutputT` upstream).
pub trait ArrowValue: Copy {
    /// upstream: `Iterator::null_default`.
    const NULL: Self;
    fn from_i64(v: i64) -> Self;
    fn from_u64(v: u64) -> Self;
    fn from_f32(v: f32) -> Self;
    fn from_f64(v: f64) -> Self;
    fn from_bool(v: bool) -> Self;
}

impl ArrowValue for f64 {
    const NULL: Self = f64::NAN;
    fn from_i64(v: i64) -> Self {
        v as f64
    }
    fn from_u64(v: u64) -> Self {
        v as f64
    }
    fn from_f32(v: f32) -> Self {
        v as f64
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn from_bool(v: bool) -> Self {
        v as u8 as f64
    }
}

impl ArrowValue for f32 {
    const NULL: Self = f32::NAN;
    fn from_i64(v: i64) -> Self {
        v as f32
    }
    fn from_u64(v: u64) -> Self {
        v as f32
    }
    fn from_f32(v: f32) -> Self {
        v
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn from_bool(v: bool) -> Self {
        v as u8 as f32
    }
}

impl ArrowValue for i32 {
    const NULL: Self = 0;
    fn from_i64(v: i64) -> Self {
        v as i32
    }
    fn from_u64(v: u64) -> Self {
        v as i32
    }
    fn from_f32(v: f32) -> Self {
        v as i32
    }
    fn from_f64(v: f64) -> Self {
        v as i32
    }
    fn from_bool(v: bool) -> Self {
        v as i32
    }
}

#[inline]
fn bit(bits: *const u8, i: usize) -> bool {
    unsafe { (*bits.add(i >> 3) >> (i & 7)) & 1 == 1 }
}

/// A typed view of one or more chunks (upstream `ArrowChunkedArray::View`).
struct View<'a> {
    ty: ArrowType,
    schema: &'a ArrowSchema,
    chunks: Vec<&'a ArrowArray>,
}

impl<'a> View<'a> {
    /// upstream: `View(std::vector<View>)`.
    fn concat(views: Vec<View<'a>>) -> Result<Self> {
        let mut it = views.into_iter();
        let mut first = it.next().ok_or_else(|| LgbmError::InvalidData("no fields to concatenate".into()))?;
        for v in it {
            if v.ty != first.ty {
                return Err(LgbmError::InvalidData(format!(
                    "All views must have the same type, but got {} and {}",
                    v.ty.name(),
                    first.ty.name()
                )));
            }
            first.chunks.extend(v.chunks);
        }
        Ok(first)
    }

    /// upstream: `View::field`.
    fn field(&self, j: usize) -> Result<View<'a>> {
        if self.ty != ArrowType::Struct {
            return Err(LgbmError::InvalidData(format!("Expected struct type for array, got {}", self.ty.name())));
        }
        let schema: &'a ArrowSchema = unsafe { &**self.schema.children.add(j) };
        let chunks = self.chunks.iter().map(|c| unsafe { &**c.children.add(j) }).collect();
        Ok(View { ty: ArrowType::of(schema)?, schema, chunks })
    }

    /// upstream: `View::visit` followed by a full iteration.
    fn values<T: ArrowValue>(&self) -> Result<Vec<T>> {
        let ty = self.ty;
        if !matches!(
            ty,
            ArrowType::Int8
                | ArrowType::Int16
                | ArrowType::Int32
                | ArrowType::Int64
                | ArrowType::UInt8
                | ArrowType::UInt16
                | ArrowType::UInt32
                | ArrowType::UInt64
                | ArrowType::Float
                | ArrowType::Double
                | ArrowType::Bool
        ) {
            return Err(LgbmError::InvalidData(format!("Unsupported Arrow type: {}", ty.name())));
        }
        let total: i64 = self.chunks.iter().map(|c| c.length).sum();
        let mut out = Vec::with_capacity(total.max(0) as usize);
        for arr in &self.chunks {
            if arr.length == 0 {
                continue;
            }
            if arr.n_buffers < 2 || arr.buffers.is_null() {
                return Err(LgbmError::InvalidData("Arrow array has fewer than 2 buffers".into()));
            }
            let validity = unsafe { *arr.buffers } as *const u8;
            let data = unsafe { *arr.buffers.add(1) };
            let offset = arr.offset as usize;
            for e in 0..arr.length as usize {
                let idx = offset + e;
                if !validity.is_null() && !bit(validity, idx) {
                    out.push(T::NULL);
                    continue;
                }
                // SAFETY: `idx < offset + length`, within the producer's buffer for this type.
                let v = unsafe {
                    match ty {
                        ArrowType::Int8 => T::from_i64(*(data as *const i8).add(idx) as i64),
                        ArrowType::Int16 => T::from_i64(*(data as *const i16).add(idx) as i64),
                        ArrowType::Int32 => T::from_i64(*(data as *const i32).add(idx) as i64),
                        ArrowType::Int64 => T::from_i64(*(data as *const i64).add(idx)),
                        ArrowType::UInt8 => T::from_u64(*(data as *const u8).add(idx) as u64),
                        ArrowType::UInt16 => T::from_u64(*(data as *const u16).add(idx) as u64),
                        ArrowType::UInt32 => T::from_u64(*(data as *const u32).add(idx) as u64),
                        ArrowType::UInt64 => T::from_u64(*(data as *const u64).add(idx)),
                        ArrowType::Float => T::from_f32(*(data as *const f32).add(idx)),
                        ArrowType::Double => T::from_f64(*(data as *const f64).add(idx)),
                        ArrowType::Bool => T::from_bool(bit(data as *const u8, idx)),
                        _ => unreachable!(),
                    }
                };
                out.push(v);
            }
        }
        Ok(out)
    }
}

/// An owned Arrow chunked array (upstream `ArrowChunkedArray`); a struct
/// type is a table with one field per column.
#[derive(Debug)]
pub struct ArrowChunkedArray {
    ty: ArrowType,
    schema: Box<ArrowSchema>,
    chunks: Vec<ArrowArray>,
}

impl Drop for ArrowChunkedArray {
    fn drop(&mut self) {
        for c in &mut self.chunks {
            if let Some(release) = c.release {
                unsafe { release(c) };
            }
        }
        if let Some(release) = self.schema.release {
            unsafe { release(&mut *self.schema) };
        }
    }
}

impl ArrowChunkedArray {
    /// Consume an `ArrowArrayStream`, taking ownership of its schema and all chunks.
    ///
    /// The stream is moved out of `*stream` (its `release` is set to null, so the
    /// producer's capsule destructor does nothing) and released here.
    ///
    /// # Safety
    /// `stream` must point to a valid, unreleased `ArrowArrayStream`.
    pub unsafe fn from_stream(stream: *mut ArrowArrayStream) -> Result<Self> {
        if stream.is_null() || unsafe { (*stream).release.is_none() } {
            return Err(LgbmError::InvalidData("Arrow array stream is null or already released".into()));
        }
        let mut owned = unsafe { ptr::read(stream) };
        unsafe { (*stream).release = None };
        let result = unsafe { Self::read_stream(&mut owned) };
        if let Some(release) = owned.release {
            unsafe { release(&mut owned) };
        }
        result
    }

    unsafe fn read_stream(stream: &mut ArrowArrayStream) -> Result<Self> {
        let last_error = |stream: &mut ArrowArrayStream| -> String {
            match stream.get_last_error {
                Some(f) => {
                    let p = unsafe { f(stream) };
                    if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
                }
                None => String::new(),
            }
        };
        let mut schema = Box::new(ArrowSchema::empty());
        let get_schema = stream.get_schema.ok_or_else(|| LgbmError::InvalidData("stream has no get_schema".into()))?;
        if unsafe { get_schema(stream, &mut *schema) } != 0 {
            return Err(LgbmError::InvalidData(format!(
                "Failed to get schema from Arrow array stream: {}",
                last_error(stream)
            )));
        }
        let mut out = Self { ty: ArrowType::Other("na"), schema, chunks: Vec::new() };
        out.ty = ArrowType::of(&out.schema)?;
        let get_next = stream.get_next.ok_or_else(|| LgbmError::InvalidData("stream has no get_next".into()))?;
        loop {
            let mut chunk = ArrowArray::empty();
            if unsafe { get_next(stream, &mut chunk) } != 0 {
                return Err(LgbmError::InvalidData(format!(
                    "Failed to get next chunk from Arrow array stream: {}",
                    last_error(stream)
                )));
            }
            if chunk.release.is_none() {
                break;
            }
            out.chunks.push(chunk);
        }
        Ok(out)
    }

    /// upstream: `is_struct`.
    pub fn is_struct(&self) -> bool {
        self.ty == ArrowType::Struct
    }

    /// Total number of elements (rows), the sum of all chunk lengths.
    pub fn len(&self) -> usize {
        self.chunks.iter().map(|c| c.length.max(0) as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// upstream: `get_num_fields`.
    pub fn num_fields(&self) -> Result<usize> {
        if !self.is_struct() {
            return Err(LgbmError::InvalidData(format!("Expected struct type for array, got {}", self.ty.name())));
        }
        Ok(self.schema.n_children.max(0) as usize)
    }

    /// Field names of a struct (table) array.
    pub fn field_names(&self) -> Result<Vec<String>> {
        (0..self.num_fields()?)
            .map(|j| {
                let child = unsafe { &**self.schema.children.add(j) };
                Ok(if child.name.is_null() {
                    String::new()
                } else {
                    unsafe { CStr::from_ptr(child.name) }.to_string_lossy().into_owned()
                })
            })
            .collect()
    }

    /// upstream: `view()` (empty chunks are skipped).
    fn view(&self) -> View<'_> {
        View { ty: self.ty, schema: &self.schema, chunks: self.chunks.iter().filter(|c| c.length != 0).collect() }
    }

    /// All values of a non-struct array, cast to `T`.
    pub fn values<T: ArrowValue>(&self) -> Result<Vec<T>> {
        self.view().values()
    }

    /// Values of field `j` of a struct (table) array, cast to `T`.
    pub fn field_values<T: ArrowValue>(&self, j: usize) -> Result<Vec<T>> {
        let n = self.num_fields()?;
        if j >= n {
            return Err(LgbmError::InvalidData(format!("field index {j} out of range for {n} fields")));
        }
        self.view().field(j)?.values()
    }

    /// The fields of a struct array concatenated in order, or the values of a
    /// non-struct array (upstream: metadata.cpp `InitScoreView`).
    pub fn concatenated_values<T: ArrowValue>(&self) -> Result<Vec<T>> {
        if !self.is_struct() {
            return self.values();
        }
        let view = self.view();
        let fields = (0..self.num_fields()?).map(|j| view.field(j)).collect::<Result<Vec<_>>>()?;
        View::concat(fields)?.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::ffi::CString;

    // Test producer: arrays and schemas are leaked, and release callbacks only mark them released.
    unsafe extern "C" fn release_schema(s: *mut ArrowSchema) {
        unsafe { (*s).release = None };
    }

    unsafe extern "C" fn release_array(a: *mut ArrowArray) {
        unsafe { (*a).release = None };
    }

    struct StreamState {
        schema: Option<ArrowSchema>,
        chunks: VecDeque<ArrowArray>,
    }

    unsafe extern "C" fn get_schema(s: *mut ArrowArrayStream, out: *mut ArrowSchema) -> c_int {
        let state = unsafe { &mut *((*s).private_data as *mut StreamState) };
        unsafe { ptr::write(out, state.schema.take().unwrap()) };
        0
    }

    unsafe extern "C" fn get_next(s: *mut ArrowArrayStream, out: *mut ArrowArray) -> c_int {
        let state = unsafe { &mut *((*s).private_data as *mut StreamState) };
        unsafe { ptr::write(out, state.chunks.pop_front().unwrap_or_else(ArrowArray::empty)) };
        0
    }

    unsafe extern "C" fn release_stream(s: *mut ArrowArrayStream) {
        unsafe { (*s).release = None };
    }

    fn cstr(s: &str) -> *const c_char {
        Box::leak(CString::new(s).unwrap().into_boxed_c_str()).as_ptr()
    }

    fn schema(format: &str, children: Vec<ArrowSchema>) -> ArrowSchema {
        let mut s = ArrowSchema::empty();
        s.format = cstr(format);
        s.name = cstr("");
        s.n_children = children.len() as i64;
        let ptrs: Vec<*mut ArrowSchema> = children.into_iter().map(|c| Box::into_raw(Box::new(c))).collect();
        s.children = Box::leak(ptrs.into_boxed_slice()).as_mut_ptr();
        s.release = Some(release_schema);
        s
    }

    fn primitive_schema(format: &str) -> ArrowSchema {
        schema(format, Vec::new())
    }

    fn struct_schema(formats: &[&str]) -> ArrowSchema {
        schema("+s", formats.iter().map(|f| primitive_schema(f)).collect())
    }

    fn bitmap(bits: impl Iterator<Item = bool>) -> Vec<u8> {
        let mut out = Vec::new();
        for (i, b) in bits.enumerate() {
            if i % 8 == 0 {
                out.push(0);
            }
            if b {
                *out.last_mut().unwrap() |= 1 << (i % 8);
            }
        }
        out
    }

    fn leak_buffers(buffers: Vec<*const c_void>) -> *mut *const c_void {
        Box::leak(buffers.into_boxed_slice()).as_mut_ptr()
    }

    /// Like upstream's `MakePrimitiveArray`: `values` with nulls at `null_indices`,
    /// then sliced by `offset` (format "f" stores f32 values, "b" stores bits).
    fn primitive_array(format: &str, values: &[f64], null_indices: &[usize], offset: i64) -> ArrowArray {
        let n = values.len();
        let validity = if null_indices.is_empty() {
            ptr::null()
        } else {
            Box::leak(bitmap((0..n).map(|i| !null_indices.contains(&i))).into_boxed_slice()).as_ptr().cast()
        };
        let data: *const c_void = match format {
            "f" => Box::leak(values.iter().map(|&v| v as f32).collect::<Vec<_>>().into_boxed_slice()).as_ptr().cast(),
            "b" => Box::leak(bitmap(values.iter().map(|&v| v != 0.0)).into_boxed_slice()).as_ptr().cast(),
            _ => unreachable!(),
        };
        let mut a = ArrowArray::empty();
        a.length = n as i64 - offset;
        a.offset = offset;
        a.null_count = null_indices.len() as i64;
        a.n_buffers = 2;
        a.buffers = leak_buffers(vec![validity, data]);
        a.release = Some(release_array);
        a
    }

    /// upstream: `MakeStream`, consumed by `ArrowChunkedArray(ArrowArrayStream*)`.
    fn chunked(schema: ArrowSchema, chunks: Vec<ArrowArray>) -> ArrowChunkedArray {
        let state = Box::new(StreamState { schema: Some(schema), chunks: chunks.into() });
        let mut stream = ArrowArrayStream {
            get_schema: Some(get_schema),
            get_next: Some(get_next),
            get_last_error: None,
            release: Some(release_stream),
            private_data: Box::into_raw(state).cast(),
        };
        let out = unsafe { ArrowChunkedArray::from_stream(&mut stream) }.unwrap();
        assert!(stream.release.is_none(), "the stream is moved out of the producer's struct");
        out
    }

    // upstream: tests/cpp_tests/test_arrow.cpp TEST(ArrowChunkedArrayTest, GetLength)
    #[test]
    fn get_length() {
        let a = chunked(primitive_schema("f"), vec![primitive_array("f", &[1.0, 2.0], &[], 0)]);
        assert_eq!(a.len(), 2);
        let a = chunked(
            primitive_schema("f"),
            vec![primitive_array("f", &[1.0, 2.0], &[], 0), primitive_array("f", &[3.0, 4.0, 5.0, 6.0], &[], 0)],
        );
        assert_eq!(a.len(), 6);
        let a = chunked(primitive_schema("b"), vec![primitive_array("b", &[1.0, 0.0, 1.0, 1.0], &[], 1)]);
        assert_eq!(a.len(), 3);
    }

    // upstream: tests/cpp_tests/test_arrow.cpp TEST(ArrowChunkedArrayTest, GetFields)
    #[test]
    fn get_fields() {
        let children = [primitive_array("f", &[1.0, 2.0, 3.0], &[], 0), primitive_array("f", &[4.0, 5.0, 6.0], &[], 0)];
        let ptrs: Vec<*mut ArrowArray> = children.into_iter().map(|c| Box::into_raw(Box::new(c))).collect();
        let mut array = ArrowArray::empty();
        array.length = 3;
        array.n_buffers = 1;
        array.buffers = leak_buffers(vec![ptr::null()]);
        array.n_children = 2;
        array.children = Box::leak(ptrs.into_boxed_slice()).as_mut_ptr();
        array.release = Some(release_array);
        let a = chunked(struct_schema(&["f", "f"]), vec![array]);
        assert_eq!(a.len(), 3);
        assert_eq!(a.num_fields().unwrap(), 2);
        assert_eq!(a.field_values::<i32>(0).unwrap()[0], 1);
        assert_eq!(a.field_values::<i32>(1).unwrap()[0], 4);
    }

    // upstream: tests/cpp_tests/test_arrow.cpp TEST(ArrowChunkedArrayTest, IteratorArithmetic)
    // Adapted: the C++ random-access iterator is internal; the same positions are
    // checked on the materialized values.
    #[test]
    fn iterator_arithmetic() {
        let a = chunked(
            primitive_schema("f"),
            vec![
                primitive_array("f", &[1.0, 2.0], &[], 0),
                primitive_array("f", &[3.0, 4.0, 5.0, 6.0], &[], 0),
                primitive_array("f", &[7.0], &[], 0),
            ],
        );
        let v = a.values::<i32>().unwrap();
        assert_eq!((v[0], v[1], v[2], v[4], v[6]), (1, 2, 3, 5, 7));
        assert_eq!(v.len(), 7);
    }

    // upstream: tests/cpp_tests/test_arrow.cpp TEST(ArrowChunkedArrayTest, BooleanIterator)
    #[test]
    fn boolean_iterator() {
        let a = chunked(
            primitive_schema("b"),
            vec![
                primitive_array("b", &[0.0, 1.0, 0.0], &[2], 0),
                primitive_array("b", &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0], &[], 1),
            ],
        );
        let v = a.values::<f32>().unwrap();
        assert_eq!((v[0], v[1]), (0.0, 1.0));
        assert!(v[2].is_nan());
        assert_eq!((v[3], v[6], v[10], v[11]), (0.0, 1.0, 0.0, 1.0));
        assert_eq!(v.len(), 12);
    }

    // upstream: tests/cpp_tests/test_arrow.cpp TEST(ArrowChunkedArrayTest, OffsetAndValidity)
    #[test]
    fn offset_and_validity() {
        let a = chunked(
            primitive_schema("f"),
            vec![primitive_array("f", &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], 2)],
        );
        let v = a.values::<f64>().unwrap();
        assert!(v[0].is_nan() && v[1].is_nan());
        // upstream's `it[c]` indexes from the start of the (single) chunk, not from `it`.
        assert_eq!((v[2], v[4]), (4.0, 6.0));
    }

    #[test]
    fn concatenates_fields_and_rejects_unsupported_types() {
        let a = chunked(primitive_schema("f"), vec![primitive_array("f", &[1.0], &[], 0)]);
        assert_eq!(a.concatenated_values::<f64>().unwrap(), vec![1.0]);
        let s = chunked(primitive_schema("u"), Vec::new());
        assert_eq!(
            s.values::<f64>().unwrap_err().to_string(),
            "invalid data: Unsupported Arrow type: string"
        );
        assert_eq!(
            s.num_fields().unwrap_err().to_string(),
            "invalid data: Expected struct type for array, got string"
        );
        let t = chunked(struct_schema(&["f", "g"]), Vec::new());
        assert_eq!(t.field_names().unwrap(), vec!["", ""]);
        assert!(t.concatenated_values::<f64>().unwrap_err().to_string().contains("All views must have the same type"));
    }
}

