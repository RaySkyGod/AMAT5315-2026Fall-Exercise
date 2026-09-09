/// Returns the greeting shown by the `md` binary.
pub fn greeting() -> String {
    "Hello, world!".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}
