pub mod io {
    use serde::Deserialize;
    use toml::de::Error;
    use toml::Value;

    // Define the Argument struct with the Deserialize trait
    #[derive(Debug, Deserialize)]
    pub struct Argument {
        pub batch_size: i64,
        pub test_batch_size: i64,
        pub epochs: i64,
        pub lr: f64,
        pub no_cuda: bool,
        pub log_interval: i64,
    }

    // Parses the Config.toml file
    pub fn read_configs(file_path: &str) -> Result<Argument, Error> {
        // Read the contents of the TOML file
        let toml_content = std::fs::read_to_string(file_path)
            .unwrap_or_else(|_| panic!("Unable to read {}", file_path));

        // Parse the TOML content into a TOML value
        let value: Value = toml::from_str(&toml_content)?;

        // Deserialize the TOML value into the Argument struct
        let argument: Argument = value.try_into()?;

        Ok(argument)
    }
}
