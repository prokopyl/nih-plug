mod context;
mod descriptor;
pub mod features;
mod wrapper;

mod draft_ext;
mod entry;

#[doc(hidden)]
pub use self::entry::NihClapPluginEntry;
#[doc(hidden)]
pub use clack_plugin::bundle::PluginEntryDescriptor;

/// Export a CLAP plugin from this library using the provided plugin type.
#[macro_export]
macro_rules! nih_export_clap {
    ($plugin_ty:ty) => {
        /// The CLAP plugin's entry point.
        #[allow(non_upper_case_globals)]
        #[allow(unsafe_code)]
        #[no_mangle]
        #[used]
        pub static clap_entry: $crate::wrapper::clap::PluginEntryDescriptor =
            $crate::wrapper::clap::NihClapPluginEntry::<$plugin_ty>::DESCRIPTOR;
    };
}
