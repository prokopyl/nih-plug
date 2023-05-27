use clack_plugin::extensions::prelude::*;
use clap_sys::ext::draft::remote_controls::*;
use std::ffi::CStr;
use std::mem::MaybeUninit;

#[repr(C)]
pub struct PluginRemoteControls(clap_plugin_remote_controls);

#[repr(C)]
pub struct HostRemoteControls(clap_host_remote_controls);

unsafe impl Extension for PluginRemoteControls {
    const IDENTIFIER: &'static CStr = CLAP_EXT_REMOTE_CONTROLS;
    type ExtensionSide = PluginExtensionSide;
}

// SAFETY: The API of this extension makes it so that the Send/Sync requirements are enforced onto
// the input handles, not on the descriptor itself.
unsafe impl Send for PluginRemoteControls {}
unsafe impl Sync for PluginRemoteControls {}

unsafe impl Extension for HostRemoteControls {
    const IDENTIFIER: &'static CStr = CLAP_EXT_REMOTE_CONTROLS;
    type ExtensionSide = HostExtensionSide;
}

// SAFETY: The API of this extension makes it so that the Send/Sync requirements are enforced onto
// the input handles, not on the descriptor itself.
unsafe impl Send for HostRemoteControls {}
unsafe impl Sync for HostRemoteControls {}

// Plugin-side of the extension (host-side is left unimplemented since nih-plug has no use for it)

pub struct RemoteControlsPageWriter<'a> {
    buf: &'a mut MaybeUninit<clap_remote_controls_page>,
    is_set: bool,
}

impl<'a> RemoteControlsPageWriter<'a> {
    #[inline]
    unsafe fn from_raw(raw: *mut clap_remote_controls_page) -> Self {
        Self {
            buf: &mut *raw.cast(),
            is_set: false,
        }
    }

    #[inline]
    pub fn write_raw(&mut self, raw: clap_remote_controls_page) {
        self.buf.write(raw);
        self.is_set = true;
    }
}

pub trait PluginRemoteControlsImpl {
    fn count(&self) -> u32;
    fn get(&self, index: u32, writer: &mut RemoteControlsPageWriter);
}

impl<'a, P: Plugin<'a>> ExtensionImplementation<P> for PluginRemoteControls
where
    P::MainThread: PluginRemoteControlsImpl,
{
    const IMPLEMENTATION: &'static Self = &PluginRemoteControls(clap_plugin_remote_controls {
        count: Some(count::<P>),
        get: Some(get::<P>),
    });
}

unsafe extern "C" fn count<'a, P: Plugin<'a>>(plugin: *const clap_plugin) -> u32
where
    P::MainThread: PluginRemoteControlsImpl,
{
    // FIXME: plugin pointer casts are needed here because of the clap-sys dependency difference.
    PluginWrapper::<P>::handle(plugin as *const _, |p| Ok(p.main_thread().as_ref().count()))
        .unwrap_or(0)
}

unsafe extern "C" fn get<'a, P: Plugin<'a>>(
    plugin: *const clap_plugin,
    index: u32,
    info: *mut clap_remote_controls_page,
) -> bool
where
    P::MainThread: PluginRemoteControlsImpl,
{
    PluginWrapper::<P>::handle(plugin as *const _, |p| {
        if info.is_null() {
            return Err(PluginWrapperError::NulPtr("clap_remote_controls_page"));
        };

        let mut writer = RemoteControlsPageWriter::from_raw(info);
        p.main_thread().as_ref().get(index, &mut writer);
        Ok(writer.is_set)
    })
    .unwrap_or(false)
}

impl HostRemoteControls {
    #[inline]
    pub fn suggest_page(&self, host: &HostMainThreadHandle, page: u32) {
        if let Some(suggest_page) = self.0.suggest_page {
            // FIXME: plugin pointer casts are needed here because of the clap-sys dependency difference.
            unsafe { suggest_page(host.as_raw() as *const _ as *const _, page) }
        }
    }

    #[inline]
    pub fn changed(&self, host: &mut HostMainThreadHandle) {
        if let Some(changed) = self.0.changed {
            unsafe { changed(host.as_raw() as *const _ as *const _) }
        }
    }
}
