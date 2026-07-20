# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import re
import typing as tp
from pathlib import Path

import mne
import pandas as pd

from neuralfetch import download
from neuralset.events import study


class Zada2025(study.Study):
    TASK: tp.ClassVar[str] = "task-podcast"
    SPACE: tp.ClassVar[str] = "space-MNI152NLin2009aSym"
    url: tp.ClassVar[str] = "https://openneuro.org/datasets/ds005574/versions/1.0.2"
    bibtex: tp.ClassVar[
        str
    ] = """
    @article {Zada2025,
        author = {Zada, Zaid and Nastase, Samuel A. and Aubrey, Bobbi and Jalon, Itamar and Michelmann, Sebastian and Wang, Haocheng and Hasenfratz, Liat and Doyle, Werner and Friedman, Daniel and Dugan, Patricia and Melloni, Lucia and Devore, Sasha and Flinker, Adeen and Devinsky, Orrin and Goldstein, Ariel and Hasson, Uri},
        title = {The Podcast ECoG dataset for modeling neural activity during natural language comprehension},
        elocation-id = {2025.02.14.638352},
        year = {2025},
        doi = {10.1101/2025.02.14.638352},
        publisher = {Cold Spring Harbor Laboratory},
        journal = {bioRxiv}
    }
    """
    doi: tp.ClassVar[str] = "doi:10.18112/openneuro.ds005574.v1.0.2"

    licence: tp.ClassVar[str] = "CC0"
    description: tp.ClassVar[str] = (
        "ECoG from 9 participants listening to a 30 minute podcast"
    )
    requirements: tp.ClassVar[tp.Tuple[str, ...]] = ("boto3",)

    def _download(self) -> None:
        raise NotImplementedError("Use Openneuro CLI to download the dataset")
        # download.Openneuro(study="ds005574", dset_dir=self.path).download()

    def iter_timelines(self) -> tp.Iterator[dict[str, tp.Any]]:
        for i in range(1, 10):
            yield dict(subject=f"sub-{i:02}")

    def _load_timeline_events(self, timeline: dict[str, tp.Any]) -> pd.DataFrame:
        subject = timeline["subject"]
        info = study.SpecialLoader(method=self._load_raw, timeline=timeline).to_json()
        ieeg_df = pd.DataFrame(
            [
                dict(
                    type="Ieeg",
                    start=0.0,
                    filepath=info,
                ),
            ]
        )
        audio_df = pd.DataFrame(
            [
                dict(
                    type="Audio",
                    start=0.0,
                    filepath=self._get_filename("audio", subject),
                ),
            ]
        )
        text_df = self._get_transcript(subject)
        # constituent_df = self._get_constituents(subject)
        # word_level_constituent_df = self._get_WordLevelconstituents(subject)
        return pd.concat([ieeg_df, audio_df, text_df])

    def _get_filename(self, filetype: str, subject: str) -> Path:
        ss_dir = self.path / "ds005574" / subject / "ieeg"
        stim_dir = self.path / "ds005574" / "stimuli"
        if filetype == "ieeg":
            filename = ss_dir / f"{subject}_{self.TASK}_ieeg.edf"
        elif filetype == "channels":
            filename = ss_dir / f"{subject}_{self.TASK}_channels.tsv"
        elif filetype == "coords":
            filename = ss_dir / f"{subject}_{self.SPACE}_electrodes.tsv"
        elif filetype == "transcript":
            filename = stim_dir / "syntactic" / "transcript_withsentence.tsv" #podcast_
        elif filetype == "constituency_parsing":
            filename = stim_dir / "syntactic" / "constituency_parsing.tsv"
        elif filetype == "wordLevelConstituency_parsing":
            filename = stim_dir / "syntactic" / "wordLevelConstituency_parsing.tsv"
        else:
            assert filetype == "audio"
            filename = stim_dir / "podcast.wav"
        if not filename.exists():
            raise ValueError(f"File missing for {filetype}: {filename}")
        return filename

    def _get_transcript(self, subject: str) -> pd.DataFrame:
        transcript_file = self._get_filename("transcript", subject)
        transcript_df = pd.read_csv(transcript_file,sep="\t")
        transcript_df["duration"] = transcript_df["end"] - transcript_df["start"]
        transcript_df = transcript_df.rename(columns={"word": "text"})
        # we drop punct part of speech:
        transcript_df = transcript_df[transcript_df["pos"] != "punct"]
        # we drop any word that is split into two by detecting coinciding start times:
        transcript_df = transcript_df.drop_duplicates(subset=["start"], keep="first")

        transcript_df["text"] = (
            transcript_df["text"]
            .str.lower()
            .apply(lambda text: re.sub(r"[^a-zA-Z]", "", text))
        )
        transcript_df.insert(0, "type", "Word")
        transcript_df["language"] = "english"
        
        return transcript_df

    # def _get_constituents(self, subject: str) -> pd.DataFrame:
    #     transcript_file = self._get_filename("constituency_parsing", subject)
    #     transcript_df = pd.read_csv(transcript_file,sep="\t")
    #     transcript_df["duration"] = transcript_df["end"] - transcript_df["start"]
    #     transcript_df.insert(0, "type", "Constituent")
    #     return transcript_df
    
    # def _get_WordLevelconstituents(self, subject: str) -> pd.DataFrame:
    #     transcript_file = self._get_filename("wordLevelConstituency_parsing", subject)
    #     transcript_df = pd.read_csv(transcript_file,sep="\t")
    #     transcript_df["duration"] = transcript_df["end"] - transcript_df["start"]
    #     transcript_df.insert(0, "type", "WordLevelConstituent")
    #     return transcript_df

    def _prepare_ieeg(
        self, ieeg_file: Path, coords_file: Path, ch_file: Path
    ) -> mne.io.Raw:
        raw = mne.io.read_raw_edf(ieeg_file)
        ch_names = raw.ch_names
        coords_df = pd.read_csv(coords_file, sep="\t").set_index("name")
        coords_df = coords_df.reindex(ch_names)
        ch_pos = dict(zip(coords_df.index, coords_df[["x", "y", "z"]].values.tolist()))
        ch_df = pd.read_csv(ch_file, sep="\t")
        if "Pulse Rate" in ch_names:
            raw = raw.drop_channels(["Pulse Rate"], on_missing="warn")
            ch_names = raw.ch_names
        #  Bad channels are those that are rejected either due to “no localization”
        #  or noisy power spectrum density
        bad_channels = list(ch_df.loc[ch_df.status == "bad", "name"])
        # Opting to set channels to "bad" instead of dropping
        raw.info["bads"] = bad_channels
        # Channel types
        # Set channel types
        # - Mark EKG channels as 'ecg'
        # - Mark DC and Pulse channels as 'misc' (to be excluded by default picks)
        # - Treat remaining invasive channels as 'ecog' (grid/strip)
        #   (If depth electrodes are present and identifiable by naming, they can be set to 'seeg'.)
        def _infer_ch_type(name: str) -> str:
            if name.startswith("DC") or name == "Pulse Rate":
                return "misc"
            if "EKG" in name:
                return "ecg"
            return "seeg"

        ch_types = {ch: _infer_ch_type(ch) for ch in ch_names}
        raw.set_channel_types(ch_types)
        # Assuming coord_frame="mni_tal" b/c coords reportedly for MNI152NLin2009aSym
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="mni_tal")
        return raw.set_montage(montage)

    def _load_raw(self, timeline: dict[str, tp.Any]) -> mne.io.Raw:
        subject = timeline["subject"]
        ieeg_file = self._get_filename("ieeg", subject)
        coords_file = self._get_filename("coords", subject)
        ch_file = self._get_filename("channels", subject)
        return self._prepare_ieeg(ieeg_file, coords_file, ch_file)
