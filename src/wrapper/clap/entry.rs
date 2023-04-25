use crate::prelude::ClapPlugin;
use crate::wrapper::clap::wrapper::ClapWrapperAudioProcessor;
use clack_plugin::factory::plugin::PluginFactory;
use clack_plugin::factory::PluginFactories;
use clack_plugin::prelude::*;
use std::ffi::CStr;
use std::marker::PhantomData;

pub struct NihClapPluginEntry<P: ClapPlugin>(PhantomData<P>);

impl<P: ClapPlugin> PluginEntry for NihClapPluginEntry<P> {
    fn init(_plugin_path: &CStr) -> bool {
        crate::wrapper::util::setup_logger();

        true
    }

    #[inline]
    fn declare_factories(builder: &mut PluginFactories) {
        builder.register::<PluginFactory, SinglePluginEntry<ClapWrapperAudioProcessor<P>>>();
    }
}
