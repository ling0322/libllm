// The MIT License (MIT)
//
// Copyright (c) 2024 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

package skill

import (
	"errors"
	"io"
	"time"

	"github.com/ling0322/libllm/go/ffmpegplugin"
)

var BlockSize = 60 * 16000 * 2

type WaveStream struct {
	reader       *ffmpegplugin.Reader
	buffer       []byte
	bufferOffset time.Duration
}

type WaveChunk struct {
	begin time.Duration
	end   time.Duration
	eof   bool
	data  []byte
}

func NewWaveStream(filename string) (*WaveStream, error) {
	reader, err := ffmpegplugin.NewReader(filename)
	if err != nil {
		return nil, err
	}

	return &WaveStream{
		reader: reader,
	}, nil
}

func durationToBytes(dur time.Duration) int {
	nsPerSample := 1000000000 / SampleRate
	nSamples := int(dur.Nanoseconds() / int64(nsPerSample))
	nBytes := nSamples * 2

	return nBytes
}

func (s *WaveStream) ensureOffset(offset time.Duration) error {
	if offset < s.bufferOffset {
		return errors.New("wave stream could not seek backward")
	}

	length := offset - s.bufferOffset
	for len(s.buffer) < durationToBytes(length) {
		b := make([]byte, BlockSize)
		n, err := s.reader.Read(b)
		if err != nil {
			return err
		}

		s.buffer = append(s.buffer, b[:n]...)
	}

	return nil
}

func (s *WaveStream) Seek(offset time.Duration) error {
	err := s.ensureOffset(offset)
	if err != nil {
		return err
	}

	forwardDuration := offset - s.bufferOffset
	forwardBytes := durationToBytes(forwardDuration)

	s.buffer = s.buffer[forwardBytes:]
	s.bufferOffset = offset

	return nil
}

func (s *WaveStream) Offset() time.Duration {
	return s.bufferOffset
}

func (s *WaveStream) ReadChunk(length time.Duration) (WaveChunk, error) {
	err := s.ensureOffset(s.bufferOffset + length)
	eof := false
	if errors.Is(err, io.EOF) {
		eof = true
		if len(s.buffer) == 0 {
			return WaveChunk{}, io.EOF
		}
	} else if err != nil {
		return WaveChunk{}, err
	}

	n := min(len(s.buffer), durationToBytes(length))
	return WaveChunk{
		begin: s.bufferOffset,
		end:   s.bufferOffset + length,
		eof:   eof,
		data:  s.buffer[:n],
	}, nil
}

func (s *WaveStream) Close() error {
	return s.reader.Close()
}
